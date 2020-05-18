import torch
from torchvision.utils import save_image
# load the models
from dcgan import Generator

num_gpu = 1 if torch.cuda.is_available() else 0

G = Generator(ngpu=1).eval()
G.load_state_dict(torch.load('netG_epoch_199.pth'))
if torch.cuda.is_available():
    G = G.cuda()
for i in range(100000):
    fixed_noise = torch.randn(1, 100, 1, 1)
    if torch.cuda.is_available():
        fixed_noise = fixed_noise.cuda()
    fake_images = G(fixed_noise)
    save_image(fake_images.detach(), 'generated_images/%d.png' % (i + 1), normalize=True)
