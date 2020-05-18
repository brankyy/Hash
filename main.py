import os
from collections import namedtuple
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as Data
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
from PIL import Image

# Hyper Parameters
EPOCH = 200
LR = 0.05

num_gpu = 1 if torch.cuda.is_available() else 0

# load the models
from dcgan import Generator, Alexnet, NNtoTrain, NNtoTrain_FC

G = Generator(ngpu=1).eval()
alexnet = Alexnet()
# nn_to_train = NNtoTrain()
nn_to_train_fc = NNtoTrain_FC()
# load weights
G.load_state_dict(torch.load('netG_epoch_199.pth'))
if torch.cuda.is_available():
    G = G.cuda()
    # alexnet.cuda()
    # nn_to_train.cuda()
    # nn_to_train_fc.cuda()

# print(G)
# print(alexnet)
# print(nn_to_train)
# print(nn_to_train_fc)
# for parameters in alexnet.parameters():
#     print(parameters)
for i in range(EPOCH):
# i = 1
    # rand_input1 = torch.randn(1, 64, 55, 55, requires_grad=True).cuda()
    # rand_input2 = torch.randn(1, 192, 27, 27, requires_grad=True).cuda()
    # rand_input3 = torch.randn(1, 384, 13, 13, requires_grad=True).cuda()
    # rand_input4 = torch.randn(1, 256, 13, 13, requires_grad=True).cuda()
    # rand_input5 = torch.randn(1, 256, 13, 13, requires_grad=True).cuda()
    # nn_inputs = namedtuple("nninputs", ['slice1', 'slice2', 'slice3', 'slice4', 'avg'])
    # nn_input = nn_inputs(rand_input1, rand_input2, rand_input3, rand_input4, rand_input5)
    # nn_input = [rand_input1, rand_input2, rand_input3, rand_input4, rand_input5]
    # for i in range(5):
    #     print(nn_input[i].shape)

    rand_input1 = torch.randn(1, 64, 55, 55, requires_grad=True).view(-1)
    rand_input2 = torch.randn(1, 192, 27, 27, requires_grad=True).view(-1)
    rand_input3 = torch.randn(1, 384, 13, 13, requires_grad=True).view(-1)
    rand_input4 = torch.randn(1, 256, 13, 13, requires_grad=True).view(-1)
    rand_input5 = torch.randn(1, 256, 13, 13, requires_grad=True).view(-1)
    input_cat = torch.cat((rand_input1, rand_input2, rand_input3, rand_input4, rand_input5), 0)



    # input_dataset = Data.Dataset(nn_input)

    opt_nn = torch.optim.Adam(nn_to_train_fc.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    # nn_to_train.eval()
    out_nn = nn_to_train_fc(input_cat)
    out_nn = torch.unsqueeze(out_nn, 0)
    out_nn = torch.unsqueeze(out_nn, 2)
    out_nn = torch.unsqueeze(out_nn, 3)
    # print(out_nn.shape)

    # batch_size = 1
    # latent_size = 100
    # fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
    # print(fixed_noise.shape)
    # if torch.cuda.is_available():
    #     fixed_noise = fixed_noise.cuda()
    # fake_images = G(fixed_noise)

    if torch.cuda.is_available():
        out_nn = out_nn.cuda()

    fake_images = G(out_nn)

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # print(fake_images.shape)

    save_image(fake_images.detach(), 'out/generated_sample_epoch_%d.png' % (i + 1), normalize=True)

    im = Image.open('out/generated_sample_epoch_%d.png' % (i + 1))
    # im.show()
    im_t = transform(im)
    batch_t = torch.unsqueeze(im_t, 0)
    # print(batch_t.shape)
    batch_t.cuda()
    alexnet.eval()
    out_alex = alexnet(batch_t)

    # for i in range(5):
    #     print(out_alex[i].shape)
    output_cat = torch.cat((out_alex[0].view(-1), out_alex[1].view(-1), out_alex[2].view(-1), out_alex[3].view(-1), out_alex[4].view(-1)), 0)
    # print(output_cat.shape)

    # loss1 = loss_func(out_alex[0].cuda(), nn_input[0])
    # loss2 = loss_func(out_alex[1].cuda(), nn_input[1])
    # loss3 = loss_func(out_alex[2].cuda(), nn_input[2])
    # loss4 = loss_func(out_alex[3].cuda(), nn_input[3])
    # loss5 = loss_func(out_alex[4].cuda(), nn_input[4])

    loss = loss_func(output_cat, input_cat)

    opt_nn.zero_grad()
    loss.backward()
    opt_nn.step()

    print('[%d/%d] Loss: %.4f'
          % (i + 1, EPOCH, loss.item()))


######################################################


# Input of NN ([1.64.55.55],
# [1,192,27,27],
# [1,384,13,13],
# [1,256,13,13],
# [1,256,13,13])
# Output of NN [1,100,1,1]

# print(im.shape)
# print(im)
