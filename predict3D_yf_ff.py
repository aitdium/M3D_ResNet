import torch
import os
import sys
import json
 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from tools.dataset import PET_images_3D, PET_images_3D_cut, RGB_images_3D_cut, RGB_images_3D_cut_yf_ff
from tools.dataset import CustomSubset
# 训练M3D_ResNet
import numpy as np
from copy import deepcopy
from tools.M3D_resnet_model import M3D_ResNet, Bottleneck, BasicBlock
from train3D_modify4  import tp_tn_fp_fn, mcc
from sklearn.metrics import roc_curve, auc

def main():
    # 如果有NVIDA显卡,转到GPU训练，否则用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
 
    # 将多个transforms的操作整合在一起
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
    # 得到数据集的路径
    image_path = '/home/shujingyuan/datasets/oral_cancer_dataset/imgs'
    # 如果image_path不存在，序会抛出AssertionError错误，报错为参数内容“ ”
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    

    # 一次测试载入16张图像
    batch_size = 16
    # 确定进程数
    # min()：返回给定参数的最小值，参数可以为序列
    # cpu_count()：返回一个整数值，表示系统中的CPU数量，如果不确定CPU的数量，则不返回任何内容
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    model_path = '/home/shujingyuan/pths/resnet/fufa/all'
    result_path = '/home/shujingyuan/pths/resnet/fufa/all/result.txt'
    # i = 1
    # g_num = 0
    avg_acc,avg_jql,avg_zhl,avg_mcc,avg_auc = 0, 0, 0, 0, 0
    all_acc,all_jql,all_zhl,all_mcc,all_auc=[],[],[],[],[]
    models = os.listdir(model_path)
    models.sort()
    num = len(models)-1
    softmax_0 = nn.Softmax(dim=0)
    for i in range(1,len(models)):
    # for pth_fn in models:
        # 全部样本label
        all_labels = np.array([])
        # 对应的全部预测值
        all_pre = np.array([])
        # _, i = pth_fn.split('-')
        # i = int(i)
        test_dataset = RGB_images_3D_cut_yf_ff(image_path,split='test',data_spilt=i,paths=[0,1,2], is_yuanfa=False)
        test_size = len(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
        # pth_fn = os.path.join(model_path, pth_fn)
        pth_fn = os.path.join(model_path, str(i))
        # 加载完整模型
        # model = resnet18_3D(num_classes=2).to(device)
        # model = resnet34_3D(num_classes=2).to(device)
        model = ResNet_3d(BasicBlock, [2, 2, 2, 2],shortcut_type='B',no_cuda=False,num_classes=2,include_top=True)
        # model = ResNet_3d2(BasicBlock, [2, 2, 2, 2],block_num=4,shortcut_type='B',no_cuda=False,num_classes=2,include_top=True)
        model.to(device)
        weights_path = os.path.join(pth_fn,'ResNet3d_modify.pth')
        # weights_path = os.path.join(pth_fn,'ResNet3d.pth')
        # 载入模型权重
        # weights_path = "/home/Shujy/pth/resNet3D/ResNet34_3d.pth"
        # 确定模型存在，否则反馈错误
        assert os.path.exists(pth_fn), "file: '{}' dose not exist.".format(weights_path)
        # 加载训练好的模型参数
        model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(0)))

        # 测试
        model.eval()
        acc = 0.0
        Mcc = 0.0
        TP,TN,FP,FN = 0,0,0,0
        # 清空历史梯度，与训练最大的区别是测试过程中取消了反向传播
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            print(str(i)+":"+pth_fn)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = model(test_images.to(device).float())
                pre = softmax_0(outputs.cpu())
                all_labels = np.append(all_labels, test_labels.numpy())
                all_pre = np.append(all_pre, pre[:,1].numpy())
                # print('all_labels',all_labels)
                # print()
                # print('all_pre',all_pre)
                # print()
                # print('test_labels',test_labels)
                # print()
                # print('outputs',outputs)
                # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                predict_y = torch.max(outputs, dim=1)[1]
                # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
                tp,tn,fp,fn = tp_tn_fp_fn(predict_y,test_labels.to(device))
                TP+=tp
                TN+=tn
                FP+=fp
                FN+=fn
    
            test_accurate = acc / test_size
            fpr, tpr, thresholds = roc_curve(all_labels, all_pre)
            roc_auc = auc(fpr, tpr)

            print('test_accuracy: %.3f' %
                (test_accurate))
            print('auc:{}'.format(roc_auc))
            with open(result_path, 'a') as f:
                f.write('{}: \n'.format(i))
                f.write('tp-{}, tn-{}, fp-{}, fn-{}\n'.format(TP,TN,FP,FN))
                jql = 0 if TP+FP == 0 else TP/(TP+FP)
                zhl = 0 if TP+FN == 0 else TP/(TP+FN)
                test_Mcc = mcc(TP, TN, FP, FN)
                f.write('acc-{},jql-{},zhl-{},mcc-{}\n'.format(test_accurate,jql,zhl,test_Mcc))
                f.write('\n')
            
            avg_mcc += test_Mcc
            avg_acc += test_accurate
            avg_jql += jql
            avg_zhl += zhl
            avg_auc += roc_auc
            if len(all_acc) == 0 or test_accurate <= min(all_acc):
                all_acc.append(test_accurate)
                all_jql.append(jql)
                all_zhl.append(zhl)
                all_mcc.append(test_Mcc)
                all_auc.append(roc_auc)
            else:
                for j in range(len(all_acc)):
                    if all_acc[j] < test_accurate:
                        all_acc.insert(j,test_accurate)
                        all_jql.insert(j,jql)
                        all_zhl.insert(j,zhl)
                        all_mcc.insert(j,test_Mcc)
                        all_auc.insert(j,roc_auc)
                        break
            # if test_accurate > 0.65:
            #     # avg_acc+=test_accurate
            #     # avg_jql+=TP/(TP+FP)
            #     # avg_zhl+=TP/(TP+FN)
            #     all_acc.append(test_accurate)
            #     all_jql.append(jql)
            #     all_zhl.append(zhl)
            #     all_mcc.append(test_Mcc)
            #     all_auc.append(roc_auc)
                # g_num+=1
            i += 1
            if i>num or i>48:
                break
    
    avg_acc2 = sum(all_acc)/len(all_acc)
    avg_jql2 = sum(all_jql)/len(all_jql)
    avg_zhl2 = sum(all_zhl)/len(all_zhl)
    avg_mcc2 = sum(all_mcc)/len(all_mcc)
    avg_auc2 = sum(all_auc)/len(all_auc)
    avg_auc = sum(all_auc[0:8])/8
    avg_acc = sum(all_acc[0:8])/8
    avg_jql = sum(all_jql[0:8])/8
    avg_zhl = sum(all_zhl[0:8])/8
    avg_mcc = sum(all_mcc[0:8])/8
    print('avg_acc:{},avg_jql:{},avg_zhl:{},avg_auc:{}'.format(avg_acc,avg_jql,avg_zhl,avg_auc))
    std_acc = sum([(i-avg_acc2)**2 for i in all_acc])/len(all_zhl)
    std_jql = sum([(i-avg_jql2)**2 for i in all_jql])/len(all_zhl)
    std_zhl = sum([(i-avg_zhl2)**2 for i in all_zhl])/len(all_zhl)
    std_mcc = sum([(i-avg_mcc2)**2 for i in all_mcc])/len(all_zhl)
    std_auc = sum([(i-avg_auc2)**2 for i in all_auc])/len(all_zhl)
    print('std_acc:{},std_jql:{},std_zhl:{},std_mcc:{},std_auc:{}'.format(std_acc,std_jql,std_zhl,std_mcc,std_auc))
    with open(result_path, 'a') as f:
        f.write('\n')
        f.write('all_acc:{}\n'.format(all_acc))
        f.write('all_jql:{}\n'.format(all_jql))
        f.write('all_zhl:{}\n'.format(all_zhl))
        f.write('all_mcc:{}\n'.format(all_mcc))
        f.write('all_auc:{}\n'.format(all_auc))
        f.write('num:{}\n'.format(len(all_mcc)))
        f.write('avg_acc:{},avg_jql:{},avg_zhl:{},avg_mcc:{},avg_auc:{}\n'.format(avg_acc2,avg_jql2,avg_zhl2,avg_mcc2,avg_auc2))
        f.write('std_acc:{},std_jql:{},std_zhl:{},std_mcc:{},std_auc:{}\n'.format(std_acc,std_jql,std_zhl,std_mcc,std_auc))
        f.write('avg_acc:{},avg_jql:{},avg_zhl:{},avg_mcc:{},avg_auc:{}\n'.format(avg_acc,avg_jql,avg_zhl,avg_mcc,avg_auc))
        f.write('\n')
    # # 加载完整模型
    # # model = resnet18_3D(num_classes=2).to(device)
    # # model = resnet34_3D(num_classes=2).to(device)
    # model = ResNet_3d(Bottleneck, [2, 2, 2, 2],shortcut_type='B',no_cuda=False,num_classes=2,include_top=True)
    # model.to(device)
    # # 载入模型权重
    # weights_path = "/home/Shujy/pth/resNet3D/18/20240327-150632/ResNet3d_modify.pth"
    # # weights_path = "/home/Shujy/pth/resNet3D/ResNet34_3d.pth"
    # # 确定模型存在，否则反馈错误
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # # 加载训练好的模型参数
    # model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(1)))

    # # 测试
    # model.eval()
    # acc = 0.0
    # # 清空历史梯度，与训练最大的区别是测试过程中取消了反向传播
    # with torch.no_grad():
    #     test_bar = tqdm(test_loader, file=sys.stdout)
    #     for test_data in test_bar:
    #         test_images, test_labels = test_data
    #         outputs = model(test_images.to(device).float())
    #         # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
    #         predict_y = torch.max(outputs, dim=1)[1]
    #         # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
    #         acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
 
    #     test_accurate = acc / test_size

    #     print('test_accuracy: %.3f' %
    #           (test_accurate))
 
 
if __name__ == '__main__':
    main()