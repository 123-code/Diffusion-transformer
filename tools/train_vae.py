import yaml 
import argspase
import parser
import torch 
import random 
import torchvision 
import os 
import numpy as np 
from tqdm import tqdm 
from model.vae import VAE 
from model.loss import LPIPS 
from model.discriminator import Discriminator 
from torch.utils.data.dataloader import DataLoader
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def train(args):
    with open(args.config_path,'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = VAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_config)
    model.to(device)

    im_dataset = CelebDataset(split='train',
                              im_path = dataset_config['im_path'], 
                              im_size = dataset_config['im_size'],
                              im_channels = dataset_config['im_channels'])
    

    data_loader = DataLoader(im_dataset,batch_size=train_config['autoencoder_batch_size'],shuffle=True)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    num_epochs = train_config['autoencoder_epochs']
    recon_criterion = torch.nn.MSELoss()
    disc_criterion = torch.nn.MSELoss()
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vae_autoencoder_ckpt_name'])):
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['vae_autoencoder_ckpt_name']),
                                         map_location=device))
        print('Loaded autoencoder from checkpoint')

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vae_discriminator_ckpt_name'])):
        discriminator.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                              train_config['vae_discriminator_ckpt_name']),
                                                 map_location=device))
        print('Loaded discriminator from checkpoint')

    optimizer_d = Adam(discriminator.parameters(),lr = train_config['autoencoder_lr'],betas=(0.5,0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    disc_step_start = train_config['disc_start']
    step_count = 0
    

    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0

    for epoch_idx in range(num_epochs):
        recon_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)
            model_output = model(im)
            output,encoder_output = model_output

            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8,im.shape[0])
                save_output = torch.clamp(output[:sample_size],-1.,1.).detach().cpu()
                save_output = ((save_output + 1)/2)
                save_input = ((im[:sample_size]+1)/2).detach().cpu()
                grid = make_grid(torch.cat([save_input,save_output],dim=0),nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'],'vae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'],'vae_autoencoder_samples'))
                img.save(os.path.join(train_config['task_name'],'vae_autoencoder_samples','current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
            recon_loss = recon_criterion(output,im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            mean,logvar = torch.chunk(encoder_output,2,dim=1)
            kl_loss = torch.mean(0.5*torch.sum(torch.exp(logvar) + mean**2 - 1 - logvar,dim=[1,2,3]))
            g_loss = recon_loss + (train_config['kl_weight'] * kl_loss / acc_steps)

            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,torch.zeros(disc_fake_pred.shape,device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,torch.ones(disc_real_pred.shape,device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()

            optimizer_d.step()
            optimizer_d.zero_grad()
            optimizer_g.step()
            optimizer_g.zero_grad()

            if len(disc_losses) > 0:
                
                print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
            else:
                 print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses)))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['vae_discriminator_ckpt_name']))
    print('Done Training...')

if __name__ == '__main__':
    parser.argparse.ArgumentParser(description='Arguments for vae training')
    parser.add_argument('--config',dest='config_path',default='config/celebhq.yaml',type=str)
    args = parser.parse_args()
    train(args)
                
    
    
