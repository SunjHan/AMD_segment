#!/usr/bin/python
#coding:utf-8
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast,Normalize
from albumentations.pytorch import ToTensor
import torchvision

from unet import Unet
from dataset import EDataset
from loss import DiceLoss
from deeplabv3_plus import DeepLabv3_plus
from utils import *
from res_Unet import *

import os
import scipy.misc
from PIL import Image
import numpy as np
     
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_data_path = '/data3/AMD/Training400/AMD_all/'#训练文件路径
train_data_Label_path = '/data3/AMD/Training400/Disc_Masks/' #mask文件路径
result_mask_path = '/data3/AMD/valid_mask_predict_resUnet_1128/' #结果存放路径
model_path = "/data3/AMD/weights_11.pth"

# 是否使用cuda
device = torch.device("cuda")

x_transforms = transforms.Compose([
    transforms.Resize((512,512),interpolation=2),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((512,512),interpolation=2),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def model2():
    model = Unet(3, 1).to(device)
    # model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True).to(device)
    # model3 = res_Unet(in_ch=3, out_ch=1).to(device)
    # model = DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True, _print=True).to(device)
    return model3


def train_model(model, criterion, optimizer, dataload, num_epochs=12):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y, _ in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    torch.save(model.state_dict(), '/data3/AMD/weights_%d.pth' % epoch)
    return model

#训练模型

def train(args):
    model = model2()

    batch_size = args.batch_size

    # criterion = DiceLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters())
    
    eye_dataset = EDataset(train_data_path,train_data_Label_path, name='train', transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(eye_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)
    ###
    valid()


#显示valid结果
def valid():
    model = model2()
    model.load_state_dict(torch.load(model_path))
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = DiceLoss()
    # if(args.action=='valid'):
    eye_dataset = EDataset(train_data_path,train_data_Label_path, name='valid', transform=x_transforms,target_transform=y_transforms)
    # else:
    #     pass
    dataloaders = DataLoader(eye_dataset, batch_size=1, shuffle=False)

    if not os.path.exists(result_mask_path):
        print("make dirs------")
        os.makedirs(result_mask_path)

    model.eval()
    with torch.no_grad():
        i=0
        epoch_loss = 0
        dt_size = len(dataloaders.dataset)
        for x, labels, x_path in dataloaders:
            i += 1
            y=model(x)
            y = sigmoid(y)
            loss = criterion(y, labels)
            epoch_loss += loss.item()
            print("%d/%d,dice_loss:%0.3f" % (i, (dt_size - 1) // dataloaders.batch_size + 1, loss.item()))

            img_y = torch.squeeze(y).numpy()
            img_y[img_y>0.5] = 255
            img_y[img_y<=0.5] = 0
            im = Image.fromarray(img_y)
            im = im.convert("L")
            img_name = str(x_path).split('.')[0]
            img_name = img_name.split('/')[-1]
            print(img_name)
            im.save(result_mask_path + img_name + ".bmp")
        print("loss:%0.3f" % (epoch_loss/i))


if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or valid")
    parse.add_argument("--batch_size", type=int, default=1)
    # parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="valid":
        valid()
    # elif args.action=="test":
    #     valid_test(args)