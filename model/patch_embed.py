import torch 
import torch.nn as nn 
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_patch_position_embedding(pos_emb_dim,grid_size,device):
    assert pos_emb_dim %4 == 0 ,"embedding de la posicion debe ser divisible por 4"
    # extraemos height y width de grid_size
    grid_size_h,grid_size_w = grid_size
    #crea un tensor con valores grid_size_h, de tipo torch.float32 
    grid_h = torch.arange(grid_size_h,dtype=torch.float32,device=device)
    grid_w = torch.arange(grid_size_w,dtype=torch.float32,device=device)
    # usamos indexing ij para matrices
    grid = torch.meshgrid(grid_h,grid_w,indexing='ij')
    # une los tensores grid_h y grid_w en un tensor de 2xHxW
    grid = torch.stack(grid,dim=0)
    # convertimmos al tensor en unidimensional y asignamos positions al primer elemento
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

#generamos el tensor de embeddings posicionales 
    factor = 10000 ** ((torch.arange(
        start = 0,
        end = pos_emb_dim // 4,
        dtype=torch.float32,
        device=device
    )/(pos_emb_dim//4)))

    #convertir a grid_h_positions en un tensor 2d de forma (N,1)
    # se repite la segunda dimension con los valores pos_emb_dim//4, la primera se manitiene 
    #dividir por factor para que los valores sean pequeños
    grid_h_emb = grid_h_positions[:,None].repeat(1,pos_emb_dim//4)/factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb),torch.cos(grid_h_emb)],dim=-1)

    grid_w_emb = grid_w_positions[:,None].repeat(1,pos_emb_dim//4)/factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb),torch.cos(grid_w_emb)],dim=-1)
    pos_emb = torch.cat([grid_h_emb,grid_w_emb],dim=-1)
    return pos_emb


class PatchEmbedding(nn.Module):
    def __init__(self,image_height,image_width,im_channels,patch_height,patch_width,hidden_size):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.im_channels = im_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        
        #dimension de cada patch es im_channels *  altura * ancho
        patch_dim = self.im_channels * self.patch_height * self.patch_width
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim,self.hidden_size)
        )
        nn.init.xavier_uniform_(self.patch_embed[0].weight)
        nn.init.constant_(self.patch_embed[0].bias,0)
    def forward(self,x):
        #calculamos cuantos parces se pueden extraer de la imagen
        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width
# la notacion (nh ph) y (nw pw) indica que la altura y ancho de la imagen se dividen en parches patch height y patch width

        out = rearrange(x,'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph = self.patch_height,
                        pw = self.patch_width)
        

        out = self.patch_embed(out)
        pos_embed = get_patch_position_embedding(pos_emb_dim = self.hidden_size,grid_size=(grid_size_h,grid_size_w),device=x.device)
        out = out + pos_embed
        return out


