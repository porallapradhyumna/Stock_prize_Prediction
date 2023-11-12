import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,n_heads, d_embed, in_proj_bais = True, out_proj_bais =True):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_heads = d_embed//n_heads
        
        
        self.q = nn.Linear(d_embed,d_embed,bias=in_proj_bais)
        self.k = nn.Linear(d_embed,d_embed,bias=in_proj_bais)
        self.v = nn.Linear(d_embed,d_embed,bias=in_proj_bais)
        
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bais)
        
    def forward(self, x,causal_mask=False):
        # x: batch_size, features, dim
        bs, seq_len, d_embed = x.shape
        intermedate_shape = (bs, seq_len,self.n_heads,self.d_heads)
        q= F.silu(self.q(x))
        k= F.silu(self.k(x))
        v= F.silu(self.v(x))
        
        q = q.view(intermedate_shape).transpose(1,2)
        k = k.view(intermedate_shape).transpose(1,2)
        v = v.view(intermedate_shape).transpose(1,2)
        
        #(Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) => (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1,-2)
        
        if causal_mask:
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.masked_fill_(mask,-torch.inf)
        
        weight /= math.sqrt(self.d_heads)
        
        weight = F.softmax(weight,dim=-1)
        #(Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) ==> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 
        
        #(Batch_Size, Seq_Len, H, Dim / H) ==> (Batch_Size, Seq_Len, Dim)
        output =  output.reshape((bs, seq_len, d_embed))
        
        output = self.out_proj(output)
        return output