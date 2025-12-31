# 将数据集按train:test:vaild=8:1:1的比例随机分割数据集
# （保证一个患者的数据不会同时在两个数据集中）
# 保证正负样本比例
import csv
import os
def train_test_vaild_spilt(csvpath, filedir):
    import random
    p_sample = []
    n_sample = []
    file_num = sum([1 for _ in os.listdir(filedir)])
    filedir = filedir + '/{}'
    filedir = filedir.format(file_num+1)
    if not os.path.isdir(filedir):
        os.mkdir(filedir)
    with open(csvpath) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in list(csv_reader)[1:]:
            if int(row[1]) == 1:
                p_sample.append(row[0])
            else:
                n_sample.append(row[0])
    p_sample_num = len(p_sample)
    n_sample_num = len(n_sample)
    p_test_num = int(0.1*p_sample_num)
    p_vaild_num = p_test_num
    p_testIndex = random.sample(range(0, p_sample_num-1), p_test_num)
    p_vaildIndex = random.sample(range(0, p_sample_num-1-p_test_num), p_vaild_num)
    n_test_num = int(0.1*n_sample_num)
    n_vaild_num = n_test_num
    n_testIndex = random.sample(range(0, n_sample_num-1), n_test_num)
    n_vaildIndex = random.sample(range(0, n_sample_num-1-n_test_num), n_vaild_num)
    print('总正样本数量：{}，总负样本数量：{}'.format(p_sample_num, n_sample_num))
    print('验证集正样本数量：{}，负样本数量：{}'.format(p_vaild_num, n_vaild_num))
    print('测试集正样本数量：{}，负样本数量：{}'.format(p_test_num, n_test_num))
    print('训练集正样本数量：{}，负样本数量：{}'.format(p_sample_num-p_vaild_num-p_test_num,n_sample_num-n_vaild_num-n_test_num))
    p_sample2, n_sample2 = [], []
    with open (os.path.join(filedir, 'test.list'), 'w') as f:
        for i in range(len(p_sample)):
            if i in p_testIndex:
                f.write(p_sample[i]+' 1\n')
            else:
                p_sample2.append(p_sample[i])
        for i in range(len(n_sample)):
            if i in n_testIndex:
                f.write(n_sample[i]+' 0\n')
            else:
                n_sample2.append(n_sample[i])
    p_sample = [i for i in p_sample2]
    n_sample = [i for i in n_sample2]
    p_sample2, n_sample2 = [], []
    with open (os.path.join(filedir, 'valid.list'), 'w') as f:
        for i in range(len(p_sample)):
            if i in p_vaildIndex:
                f.write(p_sample[i]+' 1\n')
            else:
                p_sample2.append(p_sample[i])
        for i in range(len(n_sample)):
            if i in n_vaildIndex:
                f.write(n_sample[i]+' 0\n')
            else:
                n_sample2.append(n_sample[i])
    with open (os.path.join(filedir, 'train.list'), 'w') as f:
        for i in p_sample2:
            f.write(i+' 1\n')
        for i in n_sample2:
            f.write(i+' 0\n')
 

if __name__ == '__main__':
    # file_dir = '/home/Shujy/codes/DTC/DTC/data/mydata/backup'
    # train_test_vaild_spilt('/home/zhaoqi/PET_CT/data/labs/label.csv', file_dir)
    yf_file_dir = '/home/Shujy/oral_cancer_dataset/data_spilt/yuanfa'
    yf_csv_path = '/home/Shujy/oral_cancer_dataset/yuanfa_label.csv'
    train_test_vaild_spilt(yf_csv_path, yf_file_dir)
    print()
    # ff_file_dir = '/home/Shujy/oral_cancer_dataset/data_spilt/fufa'
    # ff_csv_path = '/home/Shujy/oral_cancer_dataset/fufa_label.csv'
    # train_test_vaild_spilt(ff_csv_path, ff_file_dir)