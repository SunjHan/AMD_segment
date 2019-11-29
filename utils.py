import torch
from PIL import Image
import os
import numpy as np



# def save_image(tensor, **para):
#     dir = 'results'
#     # loader使用torchvision中自带的transforms函数
#     loader = transforms.Compose([
#     transforms.ToTensor()])  
#     unloader = transforms.ToPILImage()
#     image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#     image = image.squeeze(0)  # remove the fake batch dimension
#     image = unloader(image)
#     if not osp.exists(dir):
#         os.makedirs(dir)
#     image.save('results_{}/s{}-c{}-l{}-e{}-sl{:4f}-cl{:4f}.jpg'
#                .format(num, para['style_weight'], para['content_weight'], para['lr'], para['epoch'],
#                        para['style_loss'], para['content_loss']))


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def dice(input_, target):
    N = target.size(0)
    smooth = 1

    input_flat = input_.view(N, -1)
    target_flat = target.view(N, -1)

    intersection = input_flat * target_flat

    score = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    
    return score

