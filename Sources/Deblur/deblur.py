from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np
import cv2
import torch.utils.data as data

from PIL import Image

from misc import *
import models.face_fed as net

from myutils.vgg16 import Vgg16
from myutils import utils
import pdb

import transforms.pix2pix as transforms

class Opt:
  def __init__ (self):
    self.dataset = 'pix2pix_val'
    self.valDataroot = './testData'
    self.mode = 'B2A'
    self.batchSize = 1
    self.valBatchSize = 1
    self.originalSize = 128
    self.imageSize = 128
    self.inputChannelSize = 3
    self.outputChannelSize = 3
    self.ngf = 64
    self.ndf = 64
    self.niter = 400
    self.lrD = 0.0002
    self.lrG = 0.0002
    self.annealStart = 0
    self.annealEvery = 400
    self.lambdaGAN = 0.01
    self.lambdaIMG = 1
    self.poolSize = 50
    self.wd = 0.0000
    self.beta1 = 0.5
    self.netG = './pretrained_models/Deblur_epoch_Best.pth'
    self.netD = ''
    self.workers = 1
    self.exp = 'sample'
    self.display = 5
    self.evalIter = 500

class imageCustomDataset(data.Dataset): 
  def __init__(self, images, transform = None, seed = None):
    self.images = images
    self.transform = transform

    if seed is not None:
      np.random.seed(seed)


  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]
    temp_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(temp_img)
    w, h = img.size
    imgA = img.crop((0, 0, w/2, h))
    imgB = img.crop((w/2, 0, w, h))

    if self.transform is not None:
          # NOTE preprocessing for each pair of images
      # imgA, imgB, imgC = self.transform(imgA, imgB, imgC)
      imgA, imgB = self.transform(img, img)
    
    return imgA, imgB

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, -1, +1)
        
    return t#norm_ip(t, t.min(), t.max())
            
def deblur_images(images):
  opt = Opt()

  create_exp_dir(opt.exp)
  opt.manualSeed = random.randint(1, 10000)
  random.seed(opt.manualSeed)
  torch.manual_seed(opt.manualSeed)
  torch.cuda.manual_seed_all(opt.manualSeed)
  print("Random Seed: ", opt.manualSeed)

  dataset = imageCustomDataset(images=images,
                              transform=transforms.Compose([
                                transforms.Scale(opt.originalSize),
                                #transforms.CenterCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ]),
                              seed=opt.manualSeed)

  valDataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=opt.valBatchSize, 
                                            shuffle=False, 
                                            num_workers=int(opt.workers))

  # get logger
  trainLogger = open('%s/train.log' % opt.exp, 'w')



  ngf = opt.ngf
  ndf = opt.ndf
  inputChannelSize = opt.inputChannelSize
  outputChannelSize= opt.outputChannelSize


  # Load Pre-trained derain model
  netS=net.Segmentation()
  netG=net.Deblur_segdl()

  #netC.apply(weights_init)


  netG.apply(weights_init)
  if opt.netG != '':
      state_dict_g = torch.load(opt.netG, map_location='cpu')
      new_state_dict_g = {}
      for k, v in state_dict_g.items():
          name = k[7:] # remove `module.`
          #print(k)
          new_state_dict_g[name] = v
      # load params
      netG.load_state_dict(new_state_dict_g)
    #netG.load_state_dict(torch.load(opt.netG))
  # print(netG)
  netG.eval()
  #netS.apply(weights_init)
  netS.load_state_dict(torch.load('./pretrained_models/SMaps_Best.pth', map_location='cpu'))
  #netS.eval()
  netS.cpu()
  netG.cpu()

  # Initialize testing data
  target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
  input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)

  val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
  val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
  label_d = torch.FloatTensor(opt.batchSize)


  target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
  input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
  depth = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
  ato = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)


  val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
  val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
  val_depth = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
  val_ato = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)


  target_128= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
  input_128 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
  target_256= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//2), (opt.imageSize//2))
  input_256 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//2), (opt.imageSize//2))

  val_target_128= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
  val_input_128 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
  val_target_256= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//2), (opt.imageSize//2))
  val_input_256 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//2), (opt.imageSize//2))

  target, input, depth, ato = target.cpu(), input.cpu(), depth.cpu(), ato.cpu()
  val_target, val_input, val_depth, val_ato = val_target.cpu(), val_input.cpu(), val_depth.cpu(), val_ato.cpu()

  target = Variable(target, volatile=True)
  input = Variable(input,volatile=True)
  depth = Variable(depth,volatile=True)
  ato = Variable(ato,volatile=True)

  target_128, input_128 = target_128.cpu(), input_128.cpu()
  val_target_128, val_input_128 = val_target_128.cpu(), val_input_128.cpu()
  target_256, input_256 = target_256.cpu(), input_256.cpu()
  val_target_256, val_input_256 = val_target_256.cpu(), val_input_256.cpu()

  target_128 = Variable(target_128)
  input_128 = Variable(input_128)
  target_256 = Variable(target_256)
  input_256 = Variable(input_256)

  label_d = Variable(label_d.cpu())

  optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)

  result_image = []

  for epoch in range(1):
    heavy, medium, light=200, 200, 200
    for i, data in enumerate(valDataloader, 0):
      if 1:
        import time
        data_val = data
        
        t0 = time.time()

        val_input_cpu, val_target_cpu = data_val

        val_target_cpu, val_input_cpu = val_target_cpu.float().cpu(), val_input_cpu.float().cpu()
        val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)

        val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
        val_target=Variable(val_target_cpu, volatile=True)


        z=0



        with torch.no_grad():
          for idx in range(val_input.size(0)):
              single_img = val_input[idx,:,:,:].unsqueeze(0)
              val_inputv = Variable(single_img, volatile=True)
              # print (val_inputv.size())
              # val_inputv = val_inputv.float().cuda()
              val_inputv_256 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.5)
              val_inputv_128 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.25)
              
              ## Get de-rained results ##
              #residual_val, x_hat_val, x_hatlv128, x_hatvl256 = netG(val_inputv, val_inputv_256, val_inputv_128)

              t1 = time.time()
              print('running time:'+str(t1 - t0))
              from PIL import Image

              #x_hat_val = netG(val_inputv)
              #smaps_vl = netS(val_inputv)
              #S_valinput = torch.cat([smaps_vl,val_inputv],1)
              """smaps,smaps64 = netS(val_inputv,val_inputv_256)
              S_input = torch.cat([smaps,val_inputv],1)
              x_hat_val, x_hat_val64 = netG(val_inputv,val_inputv_256,smaps,smaps64)"""
              
              
              #x_hatcls1,x_hatcls2,x_hatcls3,x_hatcls4,x_lst1,x_lst2,x_lst3,x_lst4 = netG(val_inputv,val_inputv_256,smaps_i,smaps_i64,class1,class2,class3,class4)
              smaps,smaps64 = netS(val_inputv,val_inputv_256)
              class1 = torch.zeros([1,1,128,128], dtype=torch.float32)
              class1[:,0,:,:] = smaps[:,0,:,:]
              class2 = torch.zeros([1,1,128,128], dtype=torch.float32)
              class2[:,0,:,:] = smaps[:,1,:,:]
              class3 = torch.zeros([1,1,128,128], dtype=torch.float32)
              class3[:,0,:,:] = smaps[:,2,:,:]
              class4 = torch.zeros([1,1,128,128], dtype=torch.float32)
              class4[:,0,:,:] = smaps[:,3,:,:]
              class_msk1 = torch.zeros([1,3,128,128], dtype=torch.float32)
              class_msk1[:,0,:,:] = smaps[:,0,:,:] 
              class_msk1[:,1,:,:] = smaps[:,0,:,:] 
              class_msk1[:,2,:,:] = smaps[:,0,:,:] 
              class_msk2 = torch.zeros([1,3,128,128], dtype=torch.float32)
              class_msk2[:,0,:,:] = smaps[:,1,:,:]
              class_msk2[:,1,:,:] = smaps[:,1,:,:]
              class_msk2[:,2,:,:] = smaps[:,1,:,:]
              class_msk3 = torch.zeros([1,3,128,128], dtype=torch.float32)
              class_msk3[:,0,:,:] = smaps[:,2,:,:] 
              class_msk3[:,1,:,:] = smaps[:,2,:,:] 
              class_msk3[:,2,:,:] = smaps[:,2,:,:] 
              class_msk4 = torch.zeros([1,3,128,128], dtype=torch.float32)
              class_msk4[:,0,:,:] = smaps[:,3,:,:]  
              class_msk4[:,1,:,:] = smaps[:,3,:,:]
              class_msk4[:,2,:,:] = smaps[:,3,:,:]
              class1 = class1.float().cpu()
              class2 = class2.float().cpu()
              class3 = class3.float().cpu()
              class4 = class4.float().cpu()
              class_msk4 = class_msk4.float().cpu()
              class_msk3 = class_msk3.float().cpu()
              class_msk2 = class_msk2.float().cpu()
              class_msk1 = class_msk1.float().cpu()
              x_hat_val, x_hat_val64,xmask1,xmask2,xmask3,xmask4,xcl_class1,xcl_class2,xcl_class3,xcl_class4 = netG(val_inputv,val_inputv_256,smaps,class1,class2,class3,class4,val_inputv,class_msk1,class_msk2,class_msk3,class_msk4)
              # x_hat1,x_hat64,xmask1,xmask2,xmask3,xmask4,xcl_class1,xcl_class2,xcl_class3,xcl_class4 = netG(input,input_256,smaps_i,class1,class2,class3,class4,target,class_msk1,class_msk2,class_msk3,class_msk4)
              #x_hat_val.data
              #val_batch_output[idx,:,:,:].copy_(x_hat_val.data[0,1,:,:])
              # print(torch.mean(xmask1),torch.mean(xmask2),torch.mean(xmask3),torch.mean(xmask4))
              print (smaps.size())
              tensor = x_hat_val.data.cpu()

              tensor = torch.squeeze(tensor)
              tensor=norm_range(tensor, None)
              # print(tensor.min(),tensor.max())

              ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
              im = Image.fromarray(ndarr)
              result_image.append(im)
              # im.save(filename)
  return result_image

