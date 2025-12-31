from PIL import Image
from torch.utils.data import Dataset
import os
import csv
import torch
import numpy as np
import pydicom
import random, math
import pickle
from dataset import RGB_images_3D_cut,RGB_images_3D_cut_yf_ff,RGB_images_2D

# 随机裁剪(单次裁剪)
# img：待剪切图像(3D or 2D)
# begin_row：剪切开始的行
# begin_column：剪切开始的列
# 在img上从begin_row行、begin_column列开始剪切一个edgexedge的图像
def cutImg(img, begin_row, begin_column, begin_dim3 = None, edge = 64, dim3_num = None):
    if begin_dim3 and dim3_num: 
        return torch.tensor(img[:,begin_dim3:begin_dim3+dim3_num,
                            begin_row:begin_row+edge,
                            begin_column:begin_column+edge])
    else:
        return torch.tensor(img[:,begin_dim3:begin_dim3+dim3_num,
                            begin_row:begin_row+edge])
    
# 判断数据增强的次数
# img_type: 有效（0），无效（1）
# dataset: 全部(1)or原发(2)or复发(3)
def enhance_num(img_type, dataset = None, split = 'train', sample_balance = True):
    p = [18,21,19]
    n = [12,9,11]
    if split == 'train' and sample_balance:
        cut = p[dataset-1] if img_type == 1 else 0
    else:
        cut = 15
    return cut

# 数据增强(2D & 3D)(先分开写)
# 输入待增强图像（tensor），输出处理后得到的多张图像（list<tensor>）
# 在随机裁剪的基础上加入噪声、翻转、旋转
# 噪声：默认none，1：高斯噪声，2：椒盐噪声
# 翻转：True or False
# 旋转：在哪个维度上旋转（1：宽x高，2：宽x深度，3：高x深度，默认1）
#       旋转角度（单位为弧度，pi）
# 可选随机裁剪的方式（是否按比例调整裁剪次数）
# dataset: 全部(1)or原发(2)or复发(3)
# 训练、测试or验证
# 参数：随机裁剪的方式、其他增强方式
# 修改随机裁剪的深度维度（最小的深度维度，确保覆盖全部切片（test、valid））
def data_enhance_3D(img, img_type, paths = [0,1,2],
                    split = 'train', sample_balance=True,
                    dataset = 1, noise = None, flip = False, 
                    rotate = None, rotate_angle = 0):
    # result_list = []
    # dim3 = 10
    if dataset not in [1,2,3]:
        print("参数 spilt 输入错误, 应为1(全部数据)、2(原发数据)或3(复发数据)")
        raise Exception
    dim3_num = 10
    _, img_edge, _, depth = img.shape
    cut_num = enhance_num(img_type, dataset, split, sample_balance)
    # 随机裁剪
    result_list = crop(img, img_type, cut_num, img_edge, depth, dim3_num, paths)
    return result_list

# 随机裁剪
def crop(img, img_type, cut_num, img_edge, img_depth = None, dim3_num = None, paths = [0,1,2]):
    results = []
    cut_rows = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
    cut_columns = [random.randint(0, img_edge - 64 - 1) for _ in range(cut_num)]
    if img_depth and dim3_num:
        cut_dim3_indexes = [random.randint(0, img_depth - dim3_num - 1) for _ in range(cut_num)]
        for index in range(cut_num):
            image = cutImg(img, cut_rows[index], cut_columns[index],
                           cut_dim3_indexes[index], 64,dim3_num)
            results.append([img_type, image[paths,:,:]])
    else:
        for index in range(cut_num):
            image = cutImg(img, cut_rows[index], cut_columns[index], edge=64)
            results.append([img_type, image[paths,:,:,:]])
    return results

# 旋转
        

# 噪声
        

# 翻转

# 样本读取
def get_img():
    pass

# 数据读取
# 2D or 3D
# 数据集划分组
# 训练、测试or验证
# 使用的通道
# 是否交叉验证
# 交叉验证的折
# 数据增强方式
def get_data(data_path, is_3D = True, data_split = 1, split='train', paths = [0,1,2], cross_validation = False, test_group = None, is_yuanfa = None):
    if is_3D:
        if is_yuanfa is None:
            return  RGB_images_3D_cut(
                        data_path=data_path, split=split,
                        data_spilt=data_split, paths=paths,
                        cross_validation=cross_validation,
                        test_group=test_group)
        else:
            return  RGB_images_3D_cut_yf_ff(
                        data_path=data_path, split=split,
                        data_spilt=data_split, paths=paths,
                        cross_validation=cross_validation,
                        test_group=test_group, is_yuanfa=is_yuanfa)
    else:
        return RGB_images_2D(
                    data_path=data_path, split=split,
                    data_spilt=data_split, paths=paths,
                    cross_validation=cross_validation,
                    test_group=test_group)

# 训练
# 不同训练策略（是否学习率递减）
# 保存模型参数策略
def model_train():
    pass


# 测试
def model_test():
    pass
