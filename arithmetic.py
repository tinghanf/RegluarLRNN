import torch
import torch.nn as nn
import argparse
from block_model import BlockModel

from torch._dynamo.utils import CompileProfiler
prof = CompileProfiler()
torch.set_float32_matmul_precision('high')

def arith_task_original(M, bs, seq_len):
    if seq_len%2==0:
        print('seq_len must be odd, reduce seq_len by 1')
        seq_len -= 1

    # (seq_len+1)//2 # of numbers for each sequence
    # (seq_len-1)//2 # of symbols for each sequence
    # (seq_len+1)//2 # of answers for each sequence
    # M, M+1, M+2 correspond to +, -, *
    nums = torch.randint(0,M, (bs, (seq_len+1)//2 ) ) 
    symb = torch.randint(M, M+3, (bs, (seq_len-1)//2))
    ans = torch.zeros(bs, (seq_len+1)//2, dtype=torch.int64)
    seqs = torch.zeros(bs, seq_len, dtype=torch.int64)

    ans[:,0] = nums[:,0]
    seqs[:,0] = nums[:,0]
    for b in range(bs):
        l, r = 0, nums[b,0]
        for i in range((seq_len-1)//2):
            if symb[b,i]==M:
                l, r = l+r, nums[b,i+1]
            elif symb[b,i]==M+1:
                l, r = l+r, -nums[b,i+1]
            else:
                r = (r * nums[b,i+1]) % M
            ans[b,i+1] = (l + r) % M
            seqs[b,2*i+1] = symb[b,i]
            seqs[b,2*i+2] = nums[b,i+1]

    return seqs, ans

class ArithMultiBlock(nn.Module):
    def __init__(self, M = 5, emb_dim=32, block_dim=2, n_layers=1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.ln_final = nn.LayerNorm(emb_dim)
        self.embedding = nn.Embedding(M+3, emb_dim)
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
parser.add_argument("--n_layers", default=3, type=int)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--bs", default=128, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
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
train_len = 40-1
test_len = 500-1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ArithMultiBlock(M, emb_dim, block_dim, n_layers=n_layers)
model = model.to(device)
print('compiling arithmetic...')
uncompiled_model = model
model = torch.compile(model, backend='inductor')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

max_acc = 0.0
for i in range(args.iters):
    train_x, train_y = arith_task_original(M, batch_size, train_len)
    train_x, train_y = train_x.to(device), train_y.to(device)
    logits = model(train_x)[:,::2]
    loss = torch.nn.functional.cross_entropy(logits.flatten(end_dim=1), train_y.flatten(end_dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1)%1000 == 0:
        test_x, test_y = arith_task_original(M, batch_size, test_len)
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
    out = open('acc_arithmetic.txt', 'a')
    out.write('{},{}\n'.format(args.seed, max_acc))
    out.flush()
