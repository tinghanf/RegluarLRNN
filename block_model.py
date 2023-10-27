import torch
import torch.nn as nn
import math

class BlockModel(nn.Module):
    def __init__(self, M = 2, emb_dim=8, block_dim=2):
        super().__init__()

        self.M = M
        self.emb_dim = emb_dim
        self.block_dim = block_dim
        h = emb_dim//block_dim

        self.gen_block = nn.Sequential(
                            nn.Linear(emb_dim, emb_dim*block_dim),
                            nn.ReLU(),
                            nn.Linear(emb_dim*block_dim, emb_dim*block_dim),
                            )
        
        self.gen_value = nn.Sequential(
                            nn.Linear(emb_dim, emb_dim),
                            nn.ReLU(),
                            nn.Linear(emb_dim, emb_dim),
                            )
        
        self.a0 = nn.Parameter( torch.randn(1, h, block_dim) )

    def get_hidden_with_v_and_pscan(self, x):
        def scan(a, As):
            c = As.shape[2]*2
            a = a.view(bs, L//c, c, h, -1)
            a1, a2 = a[:,:,:c//2], a[:,:,c//2:]

            # a2.shape = (bs, group nums, group size, h, block_dim)
            # As.shape = (bs, group nums*2-1, group size, h, block_dim, block_dim)

            assert As.shape[1]%2==1, 'works when As.shape[1]== 2**k -1 for k>=1'
            coef = As[:,::2]
            remain = As[:,1::2]
            prodd = torch.einsum('bnchij,bnhjk->bnchik', coef[:,1:], remain[:,:,-1])
            remain = torch.cat([remain, prodd], dim=2)

            # coef.shape = (bs, group nums, group size, h, block_dim, block_dim)
            # apply a group of matrix (e.g., ['A2' 'A3A2']) to the last element of a2 in each group,
            # and add together
            a2 = a2 + torch.einsum('bnchij,bnhj->bnchi', coef, a1[:,:,-1])
            a = torch.cat([a1, a2], dim=2)

            return a, remain

        bs, seq_len = x.shape[0], x.shape[1]
        h = self.emb_dim//self.block_dim
        blocks = self.gen_block(x)
        blocks = blocks.view(bs, seq_len, h, self.block_dim, self.block_dim)
        blocks = blocks - blocks.mean(dim=-2, keepdim=True)
        #blocks = blocks / (blocks.norm(dim=-2, keepdim=True).max(dim=-1, keepdim=True)[0])
        blocks = blocks / (blocks.norm(dim=-2, p=1.2, keepdim=True).max(dim=-1, keepdim=True)[0])
        
        v = self.gen_value(x).view(bs, seq_len, h, self.block_dim)

        log2_L = int(math.ceil(math.log2(seq_len+1)))
        L = 2**log2_L # the length after zero padding
        n_zero = L - seq_len - 1
        ev = torch.cat([self.a0.expand(bs,-1,-1)[:,None,:,:], v], dim=1)
        ev = nn.functional.pad(ev, (0,0,0,0,0, n_zero))
        a = ev
        As = nn.functional.pad(blocks, (0,0,0,0,0,0,0, n_zero))[:,:,None,:,:,:]
        # a.shape = (bs, L, h, block_dim)
        # As.shape = (bs, L-1, 1, h, block_dim, block_dim)

        for i in range(log2_L):
            a, As = scan(a, As)
        a = a.view(bs, L, self.emb_dim)[:,1:seq_len+1]

        return a
        
