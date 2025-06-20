import torch 
import torch.nn as nn
from model.patch_embed import PatchEmbedding
from model.transformer_layer import TransformerLayer
from einops import rearrange

def get_time_embedding(time_steps,temb_dim):
    '''
    convierte el tensor time steps en un embedding 
    usando la formula . retorna un embedding BXD de B timesteps y D dimensiones
    '''
    assert temb_dim % 2 == 0,"time embedding dimension must be divisible by 2"

    factor = 1000 ** ((torch.arange(
        start = 0,
        end=temb_dim//2,
        dtype=torch.float32,
        device=time_steps.device
    )/(temb_dim//2)))

#reshape de dimension B a B,1, luego en la nueva dimension agregamos temb_dim//2
    t_emb = time_steps[:,None].repeat(1,temb_dim//2)/factor
    t_emb = torch.cat([torch.sin(t_emb),torch.cos(t_emb)],dim=-1)
    return t_emb

class DIT(nn.Module):
    def __init__(self,im_size,im_channels,config):
        super().__init__()
        num_layers = config['num_layers']
        self.image_height = im_size
        self.image_width = im_size
        self.im_channels = im_channels
        self.hidden_size = config['hidden_size']
        self.patch_height = config['patch_size']
        self.patch_width = config['patch_size']

        self.timestep_emb_dim = config['timestep_emb_dim']
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width 
        self.patch_embed_layer = PatchEmbedding(
            image_height = self.image_height,
            image_width = self.image_width,
            im_channels = self.im_channels,
            patch_height = self.patch_height,
            patch_width = self.patch_width,
            hidden_size = self.hidden_size
        )
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestamp_emb_dim,self.hidden_size),
            nn.SiLU(),
         nn.Linear(self.hidden_size,self.hidden_size))
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_size,elementwise_affine=False,eps=1E-6)
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size,2*self.hidden_size,bias=True)
        )
        self.proj_out = nn.Linear(self.hidden_size,self.patch_height *  self.patch_width *  self.im_channels)
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self,x,t):
        out = self.patch_embed_layer(x)
        t_emb = get_time_embedding(torch.as_tensor(t).long(),self.timestep_emb_dim)
        t_emb = self.t_proj(t_emb)
        for layer in self.layers:
            out = layer(out,t_emb)
        #shif y scaling para normalizacion de outputs
        pre_mlp_shift,pre_mlp_scale = self.adaptive_norm_layer(t_emb).chunk(2,dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))
        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        return out
