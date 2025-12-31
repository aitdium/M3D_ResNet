from PIL import Image
from torch.utils.data import Dataset
import os
import csv
import torch
import numpy as np
import pydicom
import random, math
import pickle
from utils import *

# img：待剪切图像
# begin_row：剪切开始的行
# begin_column：剪切开始的列
# 在img上从begin_row行、begin_column列开始剪切一个112x112的图像
# def cutImg(img, begin_row, begin_column, begin_dim3, edge, dim3_num):
#     return torch.tensor(img[:,begin_dim3:begin_dim3+dim3_num,begin_row:begin_row+edge,begin_column:begin_column+edge])

class PET_images_3D(Dataset):
    # 一个患者的全部PET图片存储为一条数据
    # 这里的样本为患者的全部PET影像（DCM文件）
    # 标注为0-有效，1-无效
    def __init__(self, data_path, split='train', transform = None, target_transform = None):
        samples = [] # 其中每个元素为[图像存储路径,label]

        labels = '/home/zhaoqi/PET_CT/data/labs/label.csv'
        if split == 'train':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/train.list'
        elif split == 'valid':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/vaild.list'
        elif split == 'test':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/test.list'
        else:
            print("参数 spilt 输入错误，应为train、valid或test")
            raise Exception
        dim3, edge= 0, 0
        with open(read_path, 'r') as f:
            samples_list = f.readlines()
        for i in samples_list:
            sample = i.split()
            # samples_list2.append([sample[0],int(sample[1])])
            spath = os.path.join(data_path, sample[0])
            spath = os.path.join(spath, 'images/PET')
            file_num = sum([1 for _ in os.listdir(spath)])
            img_path = os.path.join(spath, os.listdir(spath)[0])
            img = pydicom.read_file(img_path).pixel_array
            img = np.array(img, dtype='uint8')
            e = img.shape[0]
            if file_num > dim3:
                dim3 = file_num
            if e > edge:
                edge = e
            samples.append([spath, int(sample[1])])
        self.samples_list = samples
        self.dim3 = dim3
        self.edge = edge
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        fn, label = self.samples_list[index] #self.imgs是一个list，self.imgs的一个元素是一个str，包含图片路径，图片标签，这些信息是在init函数中从txt文件中读取的
        # fn是一个图片路径
        img3d2 = np.zeros((self.dim3,self.edge,self.edge))

        slices = [pydicom.read_file(fn + '/' + s) for s in os.listdir(fn)]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        i = 0
        for s in slices:
            image = s.pixel_array
            image = np.array(image, dtype='uint8')
            edge = image.shape[0]
            if edge != self.edge:
                k = int(( self.edge - edge) / 2)
            else:
                k = 0
            image2 = np.zeros((self.edge,self.edge))
            image2[k:k+edge,k:k+edge] = image.copy()
            image2[0:k,:] = np.zeros((k,self.edge))
            image2[k+edge:self.edge,:] = np.zeros((k,self.edge))
            image2[:,0:k] = np.zeros((self.edge,k))
            image2[:,k+edge:self.edge] = np.zeros((self.edge,k))
            img3d2[i,:,:] = image2
            i = i + 1
        img3d = torch.from_numpy(img3d2)
            
        if self.transform is not None:
            img3d = self.transform(img3d) 
        return img3d, label
    
    def __len__(self):
        return len(self.samples_list)
    

class PET_images_3D_cut(Dataset):
    # 一个患者的全部PET图片存储为一条数据
    # 这里的样本为患者的全部PET影像（DCM文件）
    # 标注为0-有效，1-无效
    def __init__(self, data_path, split='train', transform = None, target_transform = None):
        samples = []

        labels = '/home/zhaoqi/PET_CT/data/labs/label.csv'
        if split == 'train':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/train.list'
        elif split == 'valid':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/vaild.list'
        elif split == 'test':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/test.list'
        else:
            print("参数 spilt 输入错误，应为train、valid或test")
            raise Exception
        
        dim3, edge= 0, 0
        with open(read_path, 'r') as f:
            samples_list = f.readlines()
        for i in samples_list:
            sample = i.split()
            # samples_list2.append([sample[0],int(sample[1])])
            spath = os.path.join(data_path, sample[0])
            spath = os.path.join(spath, 'images/PET')
            slices = [pydicom.read_file(os.path.join(spath, s)) for s in os.listdir(spath)]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            j = 0
            img3d2 = []
            for s in slices:
                image = s.pixel_array
                image = np.array(image, dtype='float')
                edge = image.shape[0]
                if len(img3d2) == 0:
                    img3d2 = np.zeros((len(slices),edge,edge))
                img3d2[j,:,:] = image
                j = j + 1
            img_num = img3d2.shape[0]
            if img_num < 80:
                k = math.ceil(80/img_num) - 1
            else:
                k = 0
            img3d = img3d2
            for _ in range(k):
                img3d = np.append(img3d,img3d2)
            img3d2 = img3d
            e = img3d2.shape[1]
            cut_num = 8
            cut_rows = random.sample(range(0,e-112),cut_num)
            cut_columns = random.sample(range(0,e-112),cut_num)
            # 存在问题：PET图像不一定有80张
            cut_dim3_indexes = [random.sample(range(0,img_num),80) for i in range(cut_num)]
            for index in range(cut_num):
                samples.append([int(sample[1]), cutImg(img3d2, cut_rows[index], cut_columns[index], cut_dim3_indexes[index], 112)])           # 选择某一列加入到data数组中          
        self.samples_list = samples
        self.dim3 = dim3
        self.edge = edge
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        label, img3d = self.samples_list[index] #self.imgs是一个list，self.imgs的一个元素是一个str，包含图片路径，图片标签，这些信息是在init函数中从txt文件中读取的
        # fn是一个图片路径
        # img3d2 = np.zeros((self.dim3,112,112))
        # img3d2 = []

        # slices = [pydicom.read_file(fn + '/' + s) for s in os.listdir(fn)]
        # slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        # i = 0
        # for s in slices:
        #     image = s.pixel_array
        #     image = np.array(image, dtype='uint8')
        #     edge = image.shape[0]
        #     if len(img3d2) == 0:
        #         img3d2 = np.zeros((self.dim3,edge,edge))
        #     img3d2[i,:,:] = image
        #     i = i + 1
        # img3d2 = cutImg(img3d2, cut_row, cut_column)
        # img3d = torch.from_numpy(img3d2)
            
        if self.transform is not None:
            img3d = self.transform(img3d) 
        return img3d, label
    
    def __len__(self):
        return len(self.samples_list)


class RGB_images_3D_cut(Dataset):
    # 读取RBG图片的三个通道（PET、CT、分割图）
    # R：CT
    # B：肿瘤分割图
    # G：PET
    # modify20240412
    # 加上了通道的选择
    # paths：list，其中元素只能为0、1、2，分别表示RBG图像的R、B、G三个通道
    def __init__(self, data_path, split='train',data_spilt=1, paths = [0,1,2], cross_validation = False, test_group = None, transform = None, target_transform = None):
        samples = []

        labels = '/home/shujingyuan/datasets/oral_cancer_dataset/label.csv'
        if not cross_validation:
            if split == 'train':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/all/{}/train.list'.format(data_spilt)
            elif split == 'valid':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/all/{}/valid.list'.format(data_spilt)
            elif split == 'test':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/all/{}/test.list'.format(data_spilt)
            else:
                print("参数 spilt 输入错误，应为train、valid或test")
                raise Exception
            with open(read_path, 'r') as f:
                samples_list = f.readlines()
        else:
            samples_list = []
            if test_group is None:
                print("缺少参数test_group")
                raise Exception
            elif test_group < 0 or test_group > 9 or test_group != int(test_group):
                print('参数test_group输入不正确，应为[0,9]范围内的整数')
                raise Exception
            else:
                save_path = '/home/shujingyuan/datasets/oral_cancer_dataset/10_cross_validation'
                save_path = os.path.join(save_path, 'cross_validation_{}.pkl'.format(data_spilt))
                with open(save_path, 'rb') as file:
                    groups = pickle.load(file)
                vaild_group = test_group+1 if test_group<9 else 0
                if split == 'train':
                    if vaild_group == 0:
                        train_groups = groups[1:9] + [groups[-1]]
                        # print('groups[1:9]:',groups[1:9])
                        # print('groups[-1]:\n',groups[-1])
                        # print('train_groups:\n',train_groups)
                        # print()
                    else:
                        train_groups = groups[0:test_group] + groups[vaild_group+1:-1]
                    # print(train_groups)
                    for group in train_groups:
                        samples_list = samples_list + group
                elif split in ['valid', 'test']:
                    samples_list = groups[vaild_group if split == 'valid' else test_group]
                    # print(samples_list)
                else:
                    print("参数 spilt 输入错误，应为train、valid或test")
                    raise Exception
        
        dim3, edge= 0, 0
        # print(samples_list)
        for i in samples_list:
            # print(i)
            if not cross_validation:
                sample = i.split()
            else:
                sample = i
            # samples_list2.append([sample[0],int(sample[1])])
            spath = os.path.join(data_path, sample[0])
            # spath = os.path.join(spath, 'RGB/Axial')
            imgCT, imgPET, imgLabel = [], [], []
            l = len(os.listdir(spath))
            for img_path in os.listdir(spath):
                img = np.asarray(Image.open(os.path.join(spath, img_path)).convert('RGB'))
                imgCT.append(img[:,:,0])
                imgPET.append(img[:,:,1])
                imgLabel.append(img[:,:,2])
                
                # for index in range(3):
                #     img3D.append(img[:,:,index])
            k = 40 // l + 1
            imgCT2, imgPET2, imgLabel2 = [], [], []
            for _ in range(k):
                imgCT2 = imgCT2 + imgCT
                imgPET2 = imgPET2 + imgPET
                imgLabel2 = imgLabel2 + imgLabel
            img3D = [imgCT2, imgPET2, imgLabel2]
            img3D = np.array(img3D)
            _,img_num,_,img_edge = img3D.shape
            
            if split == 'train':
                cut_num = 25 if int(sample[1]) == 0 else 38
                cut_rows = [random.randint(0, img_edge-64-1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge-64-1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0,img_num-40-1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index],64,40)
                    samples.append([int(sample[1]),image[paths,:,:,:]])
            else:
                cut_num = 20
                cut_rows = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0, img_num - 40 - 1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index], 64,40)
                    samples.append([int(sample[1]), image[paths, :, :, :]])
            # samples.append([int(sample[1]),img3D])
            # samples.append([int(sample[1]),img3D])
        self.samples_list = samples
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        label, img3d = self.samples_list[index] #self.imgs是一个list，self.imgs的一个元素是一个str，包含图片路径，图片标签，这些信息是在init函数中从txt文件中读取的            
        if self.transform is not None:
            img3d = self.transform(img3d) 
        return img3d, label
    
    def __len__(self):
        return len(self.samples_list)
    
class RGB_images_3D_cut_yf_ff(Dataset):
    # 读取RBG图片的三个通道（PET、CT、分割图）
    # R：CT
    # B：肿瘤分割图
    # G：PET
    # modify20240412
    # 加上了通道的选择
    # paths：list，其中元素只能为0、1、2，分别表示RBG图像的R、B、G三个通道
    def __init__(self, data_path, split='train', data_spilt=1, paths=[0, 1, 2], is_yuanfa = True, cross_validation = False, test_group = None, transform=None, target_transform=None):
        samples = []
        if is_yuanfa:
            flag = 'yuanfa'
            p_cut = 40
            n_cut = 17
            dim3_num = 35
        else:
            flag = 'fufa'
            p_cut = 35
            n_cut = 21
            dim3_num = 53
        labels = '/home/Shujy/oral_cancer_dataset/{}_label.csv'.format(flag)
        if not cross_validation:
            if split == 'train':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/{}/{}/train.list'.format(flag, data_spilt)
            elif split == 'valid':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/{}/{}/valid.list'.format(flag, data_spilt)
            elif split == 'test':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/{}/{}/test.list'.format(flag, data_spilt)
            else:
                print("参数 spilt 输入错误，应为train、valid或test")
                raise Exception
            with open(read_path, 'r') as f:
                samples_list = f.readlines()
        else:
            samples_list = []
            if test_group is None:
                print("缺少参数test_group")
                raise Exception
            elif test_group < 0 or test_group > 9 or test_group != int(test_group):
                print('参数test_group输入不正确，应为[0,9]范围内的整数')
                raise Exception
            else:
                save_path = '/home/shujingyuan/datasets/oral_cancer_dataset/10_cross_validation_{}'.format(flag)
                save_path = os.path.join(save_path, 'cross_validation_{}.pkl'.format(data_spilt))
                with open(save_path, 'rb') as file:
                    groups = pickle.load(file)
                vaild_group = test_group+1 if test_group<9 else 0
                if split == 'train':
                    if vaild_group == 0:
                        train_groups = groups[1:9] + [groups[-1]]
                    else:
                        train_groups = groups[0:test_group] + groups[vaild_group+1:-1]
                    # print(train_groups)
                    for group in train_groups:
                        samples_list = samples_list + group
                elif split in ['valid', 'test']:
                    samples_list = groups[vaild_group if split == 'valid' else test_group]
                    # print(samples_list)
                else:
                    print("参数 spilt 输入错误，应为train、valid或test")
                    raise Exception

        dim3, edge = 0, 0
        for i in samples_list:
            if not cross_validation:
                sample = i.split()
            else:
                sample = i
            # samples_list2.append([sample[0],int(sample[1])])
            spath = os.path.join(data_path, sample[0])
            # spath = os.path.join(spath, 'RGB/Axial')
            imgCT, imgPET, imgLabel = [], [], []
            l = len(os.listdir(spath))
            for img_path in os.listdir(spath):
                img = np.asarray(Image.open(os.path.join(spath, img_path)).convert('RGB'))
                imgCT.append(img[:, :, 0])
                imgPET.append(img[:, :, 1])
                imgLabel.append(img[:, :, 2])

                # for index in range(3):
                #     img3D.append(img[:,:,index])
            k = dim3_num // l + 1
            imgCT2, imgPET2, imgLabel2 = [], [], []
            for _ in range(k):
                imgCT2 = imgCT2 + imgCT
                imgPET2 = imgPET2 + imgPET
                imgLabel2 = imgLabel2 + imgLabel
            img3D = [imgCT2, imgPET2, imgLabel2]
            img3D = np.array(img3D)
            _, img_num, _, img_edge = img3D.shape
            if split == 'train':
                cut_num = n_cut if int(sample[1]) == 0 else p_cut
                cut_rows = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0, img_num - dim3_num - 1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index], 64,dim3_num)
                    samples.append([int(sample[1]), image[paths, :, :, :]])
            else:
                cut_num = 20
                cut_rows = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0, img_num - dim3_num - 1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index], 64,dim3_num)
                    samples.append([int(sample[1]), image[paths, :, :, :]]) 
            # samples.append([int(sample[1]),img3D])
            # samples.append([int(sample[1]),img3D])
        self.samples_list = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        label, img3d = self.samples_list[
            index]  # self.imgs是一个list，self.imgs的一个元素是一个str，包含图片路径，图片标签，这些信息是在init函数中从txt文件中读取的
        if self.transform is not None:
            img3d = self.transform(img3d)
        return img3d, label

    def __len__(self):
        return len(self.samples_list)

class RGB_images_2D(Dataset):
    # 读取RBG图片的三个通道（PET、CT、分割图）
    # R：CT
    # B：肿瘤分割图
    # G：PET
    # 加上了通道的选择
    # paths：list，其中元素只能为0、1、2，分别表示RBG图像的R、B、G三个通道
    def __init__(self, data_path, split='train',data_spilt=1, paths = [0,1,2], transform = None, target_transform = None):
        samples = []

        labels = '/home/zhaoqi/PET_CT/data/labs/label.csv'
        if split == 'train':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/backup/{}/train.list'.format(data_spilt)
        elif split == 'valid':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/backup/{}/valid.list'.format(data_spilt)
        elif split == 'test':
            read_path = '/home/Shujy/codes/DTC/DTC/data/mydata/backup/{}/test.list'.format(data_spilt)
        else:
            print("参数 spilt 输入错误，应为train、valid或test")
            raise Exception
        
        self.samples_list = []
        
        with open(read_path, 'r') as f:
            samples_list = f.readlines()
        for i in samples_list:
            sample = i.split()
            # samples_list2.append([sample[0],int(sample[1])])
            spath = os.path.join(data_path, sample[0])
            # spath = os.path.join(spath, 'RGB/Axial')
            l = len(os.listdir(spath))
            for img_path in os.listdir(spath):
                img = np.asarray(Image.open(os.path.join(spath, img_path)).convert('RGB'))
                img = img.transpose(2,0,1)
                if split == 'train':
                    if int(sample[1]) == 1:
                        cut_num = 6
                    else:
                        cut_num = 4
                    cut_col = [random.randint(0,15) for _ in range(cut_num)]
                    cut_row = [random.randint(0,15) for _ in range(cut_num)]
                else:
                    cut_num = 5
                    cut_col = [0, 0, 7, 15, 15]
                    cut_row = [0, 15, 7, 0, 15]
                for cut in range(cut_num):
                    img2D = img[:, cut_row[cut]:cut_row[cut]+64, cut_col[cut]:cut_col[cut]+64]
                    self.samples_list.append([int(sample[1]), img2D, sample[0]])
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        label, img2d, name = self.samples_list[index] #self.imgs是一个list，self.imgs的一个元素是一个str，包含图片路径，图片标签，这些信息是在init函数中从txt文件中读取的            
        if self.transform is not None:
            img2d = self.transform(img2d) 
        return img2d, label, name
    
    def __len__(self):
        return len(self.samples_list)

class RGB_images_3D_cut_DA(Dataset):
    # 读取RBG图片的三个通道（PET、CT、分割图）
    # R：CT
    # B：肿瘤分割图
    # G：PET
    # modify20240412
    # 加上了通道的选择
    # paths：list，其中元素只能为0、1、2，分别表示RBG图像的R、B、G三个通道
    def __init__(self, data_path, split='train',data_spilt=1, paths = [0,1,2], cross_validation = False, test_group = None, transform = None, target_transform = None):
        samples = []

        labels = '/home/zhaoqi/PET_CT/data/labs/label.csv'
        if not cross_validation:
            if split == 'train':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/all/{}/train.list'.format(data_spilt)
            elif split == 'valid':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/all/{}/valid.list'.format(data_spilt)
            elif split == 'test':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/all/{}/test.list'.format(data_spilt)
            else:
                print("参数 spilt 输入错误，应为train、valid或test")
                raise Exception
            with open(read_path, 'r') as f:
                samples_list = f.readlines()
        else:
            samples_list = []
            if test_group is None:
                print("缺少参数test_group")
                raise Exception
            elif test_group < 0 or test_group > 9 or test_group != int(test_group):
                print('参数test_group输入不正确，应为[0,9]范围内的整数')
                raise Exception
            else:
                save_path = '/home/shujingyuan/datasets/oral_cancer_dataset/10_cross_validation'
                save_path = os.path.join(save_path, 'cross_validation_{}.pkl'.format(data_spilt))
                with open(save_path, 'rb') as file:
                    groups = pickle.load(file)
                vaild_group = test_group+1 if test_group<9 else 0
                if split == 'train':
                    if vaild_group == 0:
                        train_groups = groups[1:10]
                    else:
                        train_groups = groups[0:test_group] + groups[vaild_group+1:-1]
                    # print(train_groups)
                    for group in train_groups:
                        samples_list = samples_list + group
                elif split in ['valid', 'test']:
                    samples_list = groups[vaild_group if split == 'valid' else test_group]
                    # print(samples_list)
                else:
                    print("参数 spilt 输入错误，应为train、valid或test")
                    raise Exception
        name_list = [item[0] for item in samples_list]
        len_name_v = len(name_list)

        dim3, edge= 0, 0
        for i in samples_list:
            if not cross_validation:
                sample = i.split()
            else:
                sample = i
            # samples_list2.append([sample[0],int(sample[1])])
            spath = os.path.join(data_path, sample[0])
            name_index = name_list.index(sample[0])
            name_vector = [1 if item == name_index else 0 for item in range(len_name_v)]
            # spath = os.path.join(spath, 'RGB/Axial')
            imgCT, imgPET, imgLabel = [], [], []
            l = len(os.listdir(spath))
            for img_path in os.listdir(spath):
                img = np.asarray(Image.open(os.path.join(spath, img_path)).convert('RGB'))
                imgCT.append(img[:,:,0])
                imgPET.append(img[:,:,1])
                imgLabel.append(img[:,:,2])
                
                # for index in range(3):
                #     img3D.append(img[:,:,index])
            k = 40 // l + 1
            imgCT2, imgPET2, imgLabel2 = [], [], []
            for _ in range(k):
                imgCT2 = imgCT2 + imgCT
                imgPET2 = imgPET2 + imgPET
                imgLabel2 = imgLabel2 + imgLabel
            img3D = [imgCT2, imgPET2, imgLabel2]
            img3D = np.array(img3D)
            _,img_num,_,img_edge = img3D.shape
            if split == 'train':
                cut_num = 25 if int(sample[1]) == 0 else 38
                cut_rows = [random.randint(0, img_edge-64-1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge-64-1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0,img_num-40-1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index],64,40)
                    samples.append([int(sample[1]),image[paths,:,:,:], name_vector])
            else:
                cut_num = 20
                cut_rows = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0, img_num - 40 - 1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index], 64,40)
                    samples.append([int(sample[1]), image[paths, :, :, :], name_vector])
            # samples.append([int(sample[1]),img3D])
            # samples.append([int(sample[1]),img3D])
        self.samples_list = samples
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        label, img3d, name_vector = self.samples_list[index] #self.imgs是一个list，self.imgs的一个元素是一个str，包含图片路径，图片标签，这些信息是在init函数中从txt文件中读取的            
        if self.transform is not None:
            img3d = self.transform(img3d) 
        return img3d, label, name_vector
    
    def __len__(self):
        return len(self.samples_list)

class RGB_images_3D_cut_yf_ff_DA(Dataset):
    # 读取RBG图片的三个通道（PET、CT、分割图）
    # R：CT
    # B：肿瘤分割图
    # G：PET
    # modify20240412
    # 加上了通道的选择
    # paths：list，其中元素只能为0、1、2，分别表示RBG图像的R、B、G三个通道
    # 两个标签：是否化疗有效，患者编号（其实该图片属于某个患者的概率会更好，但参数会不会过多？）（二维向量或104维向量）
    def __init__(self, data_path, split='train', data_spilt=1, paths=[0, 1, 2], is_yuanfa = True, cross_validation = False, test_group = None, transform=None, target_transform=None):
        samples = []
        if is_yuanfa:
            flag = 'yuanfa'
            p_cut = 40
            n_cut = 17
            dim3_num = 35
        else:
            flag = 'fufa'
            p_cut = 35
            n_cut = 21
            dim3_num = 53
        labels = '/home/Shujy/oral_cancer_dataset/{}_label.csv'.format(flag)
        if not cross_validation:
            if split == 'train':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/{}/{}/train.list'.format(flag, data_spilt)
            elif split == 'valid':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/{}/{}/valid.list'.format(flag, data_spilt)
            elif split == 'test':
                read_path = '/home/shujingyuan/datasets/oral_cancer_dataset/data_spilt/{}/{}/test.list'.format(flag, data_spilt)
            else:
                print("参数 spilt 输入错误，应为train、valid或test")
                raise Exception
            with open(read_path, 'r') as f:
                samples_list = f.readlines()
        else:
            samples_list = []
            if test_group is None:
                print("缺少参数test_group")
                raise Exception
            elif test_group < 0 or test_group > 9 or test_group != int(test_group):
                print('参数test_group输入不正确，应为[0,9]范围内的整数')
                raise Exception
            else:
                save_path = '/home/shujingyuan/datasets/oral_cancer_dataset/10_cross_validation_{}'.format(flag)
                save_path = os.path.join(save_path, 'cross_validation_{}.pkl'.format(data_spilt))
                with open(save_path, 'rb') as file:
                    groups = pickle.load(file)
                vaild_group = test_group+1 if test_group<9 else 0
                if split == 'train':
                    if vaild_group == 0:
                        train_groups = groups[1:10]
                    else:
                        train_groups = groups[0:test_group] + groups[vaild_group+1:-1]
                    # print(train_groups)
                    for group in train_groups:
                        samples_list = samples_list + group
                elif split in ['valid', 'test']:
                    samples_list = groups[vaild_group if split == 'valid' else test_group]
                    # print(samples_list)
                else:
                    print("参数 spilt 输入错误，应为train、valid或test")
                    raise Exception
        name_list = [item[0] for item in samples_list]
        len_name_v = len(name_list)

        dim3, edge = 0, 0
        for i in samples_list:
            if not cross_validation:
                sample = i.split()
            else:
                sample = i
            # samples_list2.append([sample[0],int(sample[1])])
            spath = os.path.join(data_path, sample[0])
            name_index = name_list.index(sample[0])
            name_vector = [1 if item == name_index else 0 for item in range(len_name_v)]
            # spath = os.path.join(spath, 'RGB/Axial')
            imgCT, imgPET, imgLabel = [], [], []
            l = len(os.listdir(spath))
            for img_path in os.listdir(spath):
                img = np.asarray(Image.open(os.path.join(spath, img_path)).convert('RGB'))
                imgCT.append(img[:, :, 0])
                imgPET.append(img[:, :, 1])
                imgLabel.append(img[:, :, 2])

                # for index in range(3):
                #     img3D.append(img[:,:,index])
            k = dim3_num // l + 1
            imgCT2, imgPET2, imgLabel2 = [], [], []
            for _ in range(k):
                imgCT2 = imgCT2 + imgCT
                imgPET2 = imgPET2 + imgPET
                imgLabel2 = imgLabel2 + imgLabel
            img3D = [imgCT2, imgPET2, imgLabel2]
            img3D = np.array(img3D)
            _, img_num, _, img_edge = img3D.shape
            if split == 'train':
                cut_num = n_cut if int(sample[1]) == 0 else p_cut
                cut_rows = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0, img_num - dim3_num - 1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index], 64,dim3_num)
                    samples.append([int(sample[1]), image[paths, :, :, :], name_vector])
            else:
                cut_num = 20
                cut_rows = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_columns = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
                cut_dim3_indexes = [random.randint(0, img_num - dim3_num - 1) for _ in range(cut_num)]
                for index in range(cut_num):
                    image = cutImg(img3D, cut_rows[index], cut_columns[index], cut_dim3_indexes[index], 64,dim3_num)
                    samples.append([int(sample[1]), image[paths, :, :, :], name_vector]) 
            # samples.append([int(sample[1]),img3D])
            # samples.append([int(sample[1]),img3D])
        self.samples_list = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        label, img3d, name_vector = self.samples_list[index]  # self.imgs是一个list，self.imgs的一个元素是一个str，包含图片路径，图片标签，这些信息是在init函数中从txt文件中读取的
        if self.transform is not None:
            img3d = self.transform(img3d)
        return img3d, label, name_vector

    def __len__(self):
        return len(self.samples_list)

from torch.utils.data import Subset

class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.samples_list = dataset.samples_list # 保留imgs属性
        self.dim3 = dataset.dim3
        self.edge = dataset.edge
        # self.classes = dataset.classes # 保留classes属性

    def __getitem__(self, idx): #同时支持索引访问操作
        x, y = self.dataset[self.samples_list[idx]]      
        return x, y 

    def __len__(self): # 同时支持取长度操作
        return len(self.indices)
