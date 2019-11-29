from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import pandas as pd


def make_dataset_train(train_data_path, train_data_Label_path):
    # imgs=[]
    # #imgs中加入有病数据的地址
    # AMD_data_path = os.path.join(train_data_path, "AMD/")
    # for image_name in os.listdir(AMD_data_path):
    #     image_name = image_name.split('.')[0]
    #     img=os.path.join(AMD_data_path,image_name)+'.jpg'
    #     mask=os.path.join(train_data_Label_path,image_name) + '.bmp'
    #     imgs.append((img,mask))
    # #imgs中加入无病数据的地址
    # nonAMD_data_path = os.path.join(train_data_path, "Non-AMD/")
    # for image_name in os.listdir(nonAMD_data_path):
    #     image_name = image_name.split('.')[0]
    #     img=os.path.join(nonAMD_data_path,image_name)+'.jpg'
    #     mask=os.path.join(train_data_Label_path,image_name) + '.bmp'
    #     imgs.append((img,mask))
    
    # #划分训练集和测试集
    # len_train = int(len(imgs) * 0.8)
    # imgs_train = imgs[:len_train]
    # imgs_valid = imgs[len_train:]
    train = pd.read_csv('/data2/sjh/AMD/train.csv')
    valid = pd.read_csv('/data2/sjh/AMD/valid.csv')
    imgs_train = []
    imgs_valid = []

    for i in range(len(train)):
        image_name = train['name'][i].split('.')[0]
        img=os.path.join(train_data_path,image_name)+'.jpg'
        mask=os.path.join(train_data_Label_path,image_name) + '.bmp'
        imgs_train.append((img,mask))

    for i in range(len(valid)):
        image_name = valid['name'][i].split('.')[0]
        img=os.path.join(train_data_path,image_name)+'.jpg'
        mask=os.path.join(train_data_Label_path,image_name) + '.bmp'
        imgs_valid.append((img,mask))

    return imgs_train, imgs_valid

# 获取测试集
# def make_dataset_test(test_data_path):
#     imgs=[]
#     for image_name in os.listdir(test_data_path):
#         image_name = image_name.split('.')[0]
#         img=os.path.join(test_data_path,image_name)+'.jpg'
#         imgs.append(img) 

class EDataset(Dataset):
    def __init__(self, train_data_path, train_data_Label_path, name,transform=None, target_transform=None):
        imgs_train, imgs_valid = make_dataset_train(train_data_path, train_data_Label_path)
        if(name == 'train'):
            self.imgs = imgs_train
        if(name == 'valid'):
            self.imgs = imgs_valid
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        # print(x_path)
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            # print("img_X: ",np.max(img_x))
            img_x = self.transform(img_x)
            # print("img_x_transform :", img_x.max())
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            # print("img_y_transform :",img_y.max())
        # img_x /= 255
        # img_y[img_y == 255] = 1
        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)