import torch
import torch.nn as nn
import argparse
from block_model import BlockModel

from torch._dynamo.utils import CompileProfiler
prof = CompileProfiler()
torch.set_float32_matmul_precision('high')

def cycle_task(M=3, bs=10, seq_len=10):
    tokens = torch.randint(0,M, (bs,seq_len))
    cycle = tokens.cumsum(dim=-1) % M
    return tokens, cycle

class CycleMultiBlock(nn.Module):
    def __init__(self, M = 5, emb_dim=32, block_dim=2, n_layers=1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.ln_final = nn.LayerNorm(emb_dim)
        self.embedding = nn.Embedding(M, emb_dim)
        for i in range(n_layers):
            self.layers.append(BlockModel(M=M, emb_dim=emb_dim, block_dim=block_dim))
            self.ln.append( nn.LayerNorm(emb_dim) )
        self.Wo1 = nn.Linear(emb_dim, emb_dim)
        self.Wo2 = nn.Linear(emb_dim, M)

    def forward(self, input_x):
        a = self.embedding(input_x)
        for i in range(len(self.layers)-1):
            out = self.layers[i].get_hidden_with_v_and_pscan(self.ln[i](a))
            a = out + a
        a = self.layers[-1].get_hidden_with_v_and_pscan(self.ln[-1](a))
        a = self.ln_final(a)
        return self.Wo2(self.Wo1(a).relu())


parser = argparse.ArgumentParser()
parser.add_argument("--M", default=5, type=int)
parser.add_argument("--emb_dim", default=64, type=int)
parser.add_argument("--block_dim", default=8, type=int)
parser.add_argument("--n_layers", default=1, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--bs", default=128, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--iters", default=40000, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument('--write_acc', type=lambda x: str(x).lower()=='true', default=False)
args = parser.parse_args()

M = args.M
emb_dim = args.emb_dim
block_dim = args.block_dim
n_layers = args.n_layers
torch.manual_seed(args.seed)
batch_size = args.bs
train_len = 40
test_len = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CycleMultiBlock(M, emb_dim, block_dim, n_layers=n_layers)
model = model.to(device)
print('compiling cycle...')
uncompiled_model = model
model = torch.compile(model, backend='inductor')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

max_acc = 0.0
for i in range(args.iters):
    train_x, train_y = cycle_task(M, batch_size, train_len)
    train_x, train_y = train_x.to(device), train_y.to(device)
    logits = model(train_x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(end_dim=1), train_y.flatten(end_dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%1000 == 0:
        test_x, test_y = cycle_task(M, batch_size, test_len)
        test_y = test_y[:,-1]
        test_x, test_y = test_x.to(device), test_y.to(device)
        with torch.no_grad():
            logits = uncompiled_model(test_x)[:,-1]
            #logits = logits.nan_to_num(nan=0.0, posinf=1e8, neginf=-1e8)
            pred = logits.argmax(dim=-1)
            acc = (pred==test_y).float().mean()
            print(i+1, acc.item(), loss.item())

        max_acc = max(acc, max_acc)
        if (1.0-max_acc)<1e-5: break
if args.write_acc:
    out = open('acc_cycle.txt', 'a')
    out.write('{},{}\n'.format(args.seed, max_acc))
    out.flush()
