import torch 
import torch.nn as nn 
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = config['head_dim']
        self.att_dim = self.n_heads *  self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size,3 * self.att_dim,bias=True)


        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim,self.hidden_size))
        
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.constant_(self.output_proj[0].bias, 0)

    def forward(self,x):
        # tamaño del batch x numero de patches x dimenion
        B,N = x.shape[:2]
        #proyeccion de q,k,v y separacion de tensores, dividimos a la ulima dimension en chunks size self.att_dim
        q,k,v = self.qkv_proj(x).split(self.att_dim,dim=-1)

        #reorganizar q,k,v para que cada head tenga su propia dimension
        q = rearrange(q,'b n (n_h h_dim) -> b n_h n h_dim',n_h = self.n_heads,h_dim=self.head_dim)
        k = rearrange(k,'b n (n_h h_dim) -> b n_h n h_dim',n_h = self.n_heads,h_dim=self.head_dim)
        v = rearrange(v,'b n (n_h h_dim) -> b n_h n h_dim',n_h = self.n_heads,h_dim=self.head_dim)

#formula del calculo de la atencion att = (q * k^t) / (sqrt(dim(k)))
        att = torch.matmul(q,k.transpose(-2,-1)) * (self.head_dim**(-0.5))
        #softmax(att)
        att = torch.nn.functional.softmax(att,dim=-1)
        #(att * v)
        out = torch.matmul(att,v)
        out = rearrange(out,'b n_h n h_dim -> b n (n_h h_dim)')
        out = self.output_proj(out)

        return out  

        
       
        
        