import torch
from torchvision.utils import save_image
# load the models
from dcgan import Generator

num_gpu = 1 if torch.cuda.is_available() else 0

G = Generator(ngpu=1).eval()
G.load_state_dict(torch.load('netG_epoch_199.pth'))
if torch.cuda.is_available():
    G = G.cuda()

fixed_noise = torch.linspace(1, 100)
fixed_noise = torch.unsqueeze(fixed_noise, 0)
fixed_noise = torch.unsqueeze(fixed_noise, 2)
fixed_noise = torch.unsqueeze(fixed_noise, 3)
if torch.cuda.is_available():
    fixed_noise = fixed_noise.cuda()
fake_images = G(fixed_noise)
save_image(fake_images.detach(), 'test.png', normalize=True)
