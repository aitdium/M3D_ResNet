# 划分k折交叉验证
# 训练：验证：测试 = 8：1：1
# pickle存储划分结果（10折+多余的部分）
# 每组前4名为化疗无效的患者，接下来6名为化疗有效的患者
# 剩余组第1名为化疗无效的患者，接下来2名为化疗有效的患者
# 共41正样本，62负样本

import csv
import os
import random
import pickle

def k_cross_validation(label_path, k, save_path):
    # 分别获取数据集中的正例和负例
    p_sample = []
    n_sample = []
    file_num = sum([1 for _ in os.listdir(save_path)])
    # save_path = save_path + '/{}'
    # save_path = save_path.format(file_num+1)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    with open(label_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in list(csv_reader)[1:]:
            if int(row[1]) == 1:
                p_sample.append([row[0],1])
            else:
                n_sample.append([row[0],0])
    # 随机打乱样本
    random.shuffle(p_sample)
    random.shuffle(n_sample)
    # 获取正例和负例的数量
    p_sample_num = len(p_sample)
    n_sample_num = len(n_sample)
    # 每一折中正例和负例的数量
    # 余数直接放进训练集
    cross_num_p = p_sample_num // 10
    cross_num_n = n_sample_num // 10
    cross_validation = []
    print(p_sample)
    print(n_sample)
    for i in range(10):
        # 获取一组数据
        group = p_sample[i*cross_num_p:(i+1)*cross_num_p] + n_sample[i*cross_num_n:(i+1)*cross_num_n]
        # 打乱（不打乱，dataLoader会打乱数据）
        # random.shuffle(group)
        cross_validation.append(group)
    group_rest = p_sample[10*cross_num_p:] + n_sample[10*cross_num_n:]
    # random.shuffle(group_rest)
    cross_validation.append(group_rest)
    print(cross_validation)
    # for item in cross_validation:
    #     p_num = sum([1 if i in p_sample else 0 for i in item])
    #     print('正样本：', p_num)
    #     n_num = sum([1 if i in n_sample else 0 for i in item])
    #     print('负样本：', n_num)
    #     print()

    save_path = os.path.join(save_path, 'cross_validation_{}.pkl'.format(file_num+1))
    with open(save_path, 'wb') as file:
        pickle.dump(cross_validation, file)

if __name__ == '__main__':
    # label_path = '/home/shujingyuan/datasets/oral_cancer_dataset/yuanfa_label.csv'
    # save_path = '/home/shujingyuan/datasets/oral_cancer_dataset/10_cross_validation_yuanfa'
    label_path = '/home/shujingyuan/datasets/oral_cancer_dataset/fufa_label.csv'
    save_path = '/home/shujingyuan/datasets/oral_cancer_dataset/10_cross_validation_fufa'
    # label_path = '/home/shujingyuan/datasets/oral_cancer_dataset/label.csv'
    # save_path = '/home/shujingyuan/datasets/oral_cancer_dataset/10_cross_validation'
    k_cross_validation(label_path, 10, save_path)
    save_path = os.path.join(save_path, 'cross_validation_{}.pkl'.format(1))
    with open(save_path, 'rb') as file:
        my_list = pickle.load(file)

    # 打印读取的List对象
    print(my_list)
    print(len(my_list))