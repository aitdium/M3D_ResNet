import torch
import os
import sys
import json
 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from tools.dataset import RGB_images_3D_cut, RGB_images_3D_cut_yf_ff
# 训练import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter 
from torch.optim.lr_scheduler import LambdaLR  
import datetime, math
from tools.M3D_resnet_model import M3D_ResNet, Bottleneck, BasicBlock
import random
# 尽量大于0.7

seed_value = 34615   # 设定随机数种子
# 2：2020， 3：1253, 4:34615(加上学习率递减：0.001)
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value) 
torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True
 
def tp_tn_fp_fn(results, labels):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(results)):
        if results[i] == labels[i]:
            if results[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if results[i] == 1:
                FP += 1
            else:
                FN += 1
    return TP, TN, FP, FN
    # return (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)

def mcc(TP, TN, FP, FN):
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) == 0:
        return 0
    else:
        return (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    for dataset in range(1,6):
        t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # 存放每一折的训练数据
        log_path = '/home/shujingyuan/logs/M3D_ResNet/cross_validation/all/lr_decrease/{}_{}'.format(t,dataset)
        # print('log_path',log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        for tg in range(10):
            w_path = os.path.join(log_path,'{}'.format(tg))
            # print()
            # print('w_path:',w_path)
            writer = SummaryWriter(w_path)
            txt_path = os.path.join(w_path,'mix.txt')
        
            # 得到数据集的路径
            image_path = '/home/shujingyuan/datasets/oral_cancer_dataset/imgs'
            # 如果image_path不存在，序会抛出AssertionError错误，报错为参数内容“ ”
            assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
            # train_dataset = MyDataset3D(image_path)
            # print(tg)
            train_dataset = RGB_images_3D_cut(image_path,split='train',data_spilt=dataset,paths=[0,1,2],cross_validation=True,test_group=tg)
            valid_dataset = RGB_images_3D_cut(image_path,split='valid',data_spilt=dataset,paths=[0,1,2],cross_validation=True,test_group=tg)
            train_size = len(train_dataset)
            valid_size = len(valid_dataset)
            # 一次训练载入16张图像
            batch_size = 16
            # 确定进程数
            # min()：返回给定参数的最小值，参数可以为序列
            # cpu_count()：返回一个整数值，表示系统中的CPU数量，如果不确定CPU的数量，则不返回任何内容
            nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
            print('Using {} dataloader workers every process'.format(nw))
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, shuffle=True,
                                                    num_workers=nw)
            # 加载测试数据集
            validate_dataset = valid_dataset
            
            # 测试集长度
            val_num = len(validate_dataset)
            validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                        batch_size=batch_size, shuffle=True,
                                                        num_workers=nw)
        
            print("using {} images for training, {} images for validation.".format(train_size,
                                                                                val_num))
        
            # 模型实例化
            # net = M3D_ResNet(Bottleneck, [3, 4, 6, 3],shortcut_type='B',no_cuda=False,num_classes=2,include_top=True)
            net = M3D_ResNet(BasicBlock, [2, 2, 2, 2],shortcut_type='B',no_cuda=False,num_classes=2,include_top=True)
            net.to(device)
        
            # 定义损失函数（交叉熵损失）
            weight = torch.tensor([62,41]).float().to(device)
            # loss_function = nn.CrossEntropyLoss()
            loss_function = nn.CrossEntropyLoss(weight, reduction="mean")

            # 迭代次数（训练次数）
            epochs = 100
        
            # 抽取模型参数
            params = [p for p in net.parameters() if p.requires_grad]

            #shuffleNet系列中使用的学习率变化策略
            lr_lambda = lambda step : (1.0-step/epochs) if step <= epochs else 0

            # 定义adam优化器
            # optimizer = optim.Adam(net.parameters(), lr=0.000001)
            optimizer = optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0001)
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
            # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
            # L1正则化系数
            lamL1 = 0

            # 用于判断最佳模型
            # best_loss = 1e8
            best_acc = 0
            last5loss = []
            # 最佳模型保存地址
            
            save_path = '/home/shujingyuan/pths/M3D_ResNet/cross_validation/all/lr_decrease/{}_1/M3D_ResNet_{}.pth'.format(dataset, tg)
            save_dir = '/home/shujingyuan/pths/M3D_ResNet/cross_validation/all/lr_decrease/{}_1'.format(dataset)
            # 100epoch后的参数
            save_path2 = '/home/shujingyuan/pths/M3D_ResNet/cross_validation/all/lr_decrease/{}_1/M3D_ResNet_{}_2.pth'.format(dataset, tg)
            # 早停原则下保存的参数
            save_path3 = '/home/shujingyuan/pths/M3D_ResNet/cross_validation/all/lr_decrease/{}_1/M3D_ResNet_{}_3.pth'.format(dataset, tg)
            # 是否已经早停
            early_stop = False

            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            train_steps = len(train_loader)
            for epoch in range(epochs):
            # for epoch in range(1):
                train_acc = 0.0
                TP, TN, FP, FN = 0, 0, 0, 0
                # 训练
                net.train()
                running_loss = 0.0
                train_bar = tqdm(train_loader, file=sys.stdout)
                with open(txt_path, 'a') as f:
                    f.write('epoch {}:\n'.format(epoch))
                for _, data in enumerate(train_bar):
                    # 前向传播
                    images = data[0]
                    labels = torch.Tensor(data[1])
                    logits = net(images.to(device).float())
                    # with open(txt_path, 'a') as f:
                    #     f.write('output:{}\n\n'.format(logits))
                    train_results = torch.max(logits, dim=1)[1]
                    # print('logits:\n',logits)
                    # print('train_results:\n',train_results)
                    train_acc += torch.eq(train_results, labels.to(device)).sum().item()
                    tp,tn,fp,fn = tp_tn_fp_fn(train_results, labels.to(device))
                    TP+=tp
                    TN+=tn
                    FP+=fp
                    FN+=fn

                    # 计算损失
                    loss = loss_function(logits, labels.to(device))
                    # loss = loss_function(logits, labels2.to(device).float())
                    # 反向传播
                    # 清空过往梯度
                    optimizer.zero_grad()
                    # 加上L1正则项
                    # re_loss = 0
                    # for name, param in net.named_parameters():
                    #     if param.requires_grad:
                    #         re_loss += torch.sum(torch.abs(param))
                    # print(re_loss)
                    # loss = loss + lamL1 * re_loss
                    # 反向传播，计算当前梯度
                    # loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
        
                    # item()：得到元素张量的元素值
                    running_loss += loss.item()
        
                    # 进度条的前缀
                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f} lr:{:.6f}".format(epoch + 1,
                                                                            epochs,
                                                                            loss,
                                                                            optimizer.param_groups[0]['lr'])
                # scheduler.step()
                train_acc = train_acc / train_size 
                writer.add_scalar('Train Loss', running_loss/train_steps, epoch)
                writer.add_scalar('Train Acc', train_acc, epoch)
                # writer_train.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train Mcc', mcc(TP, TN, FP, FN), epoch)
                # 测试
                net.eval()
                acc = 0.0
                vTP, vTN, vFP, vFN = 0,0,0,0
                # 清空历史梯度，与训练最大的区别是测试过程中取消了反向传播
                with torch.no_grad():
                    val_bar = tqdm(validate_loader, file=sys.stdout)
                    val_loss = 0.0
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        outputs = net(val_images.to(device).float())
                        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                        predict_y = torch.max(outputs, dim=1)[1]
                        # predict_y = predict_y.numpy()
                        # predict_y = torch.FloatTensor(predict_y)
                        # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                        vtp,vtn,vfp,vfn = tp_tn_fp_fn(predict_y,val_labels.to(device))
                        val_loss += loss_function(outputs, val_labels.to(device))
                        # val_loss += loss_function(outputs, val_labels2.to(device))
                        val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                                    epochs)
                        vTP += vtp
                        vTN += vtn
                        vFP += vfp
                        vFN += vfn
            
                val_accurate = acc / valid_size
                val_mcc = mcc(vTP, vTN, vFP, vFN)
                val_loss = val_loss / len(validate_loader)
                writer.add_scalar('Valid Acc', val_accurate, epoch)
                writer.add_scalar('Valid Mcc', val_mcc, epoch)
                writer.add_scalar('Valid Loss', val_loss, epoch)
                with open(txt_path, 'a') as f:
                    f.write('\n')
                    f.write('train: tp-{}, tn-{}, fp-{}, fn-{}\n'.format(TP,TN,FP,FN))
                    f.write('valid: tp-{}, tn-{}, fp-{}, fn-{}\n'.format(vTP,vTN,vFP,vFN))
                    f.write('\n')
                print('epoch {}:'.format(epoch))
                print('train: tp-{}, tn-{}, fp-{}, fn-{}'.format(TP,TN,FP,FN))
                print('valid: tp-{}, tn-{}, fp-{}, fn-{}'.format(vTP,vTN,vFP,vFN))
                print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                    (epoch + 1, running_loss / train_steps, val_accurate))
                # 保存最好的模型权重
                # if val_loss < best_loss:
                #     best_loss = val_loss
                #     if not os.path.isdir(save_dir):
                #         os.mkdir(save_dir)
                #     torch.save(net.state_dict(), save_path)
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    # if not os.path.isdir(save_dir):
                    #     os.mkdir(save_dir)
                    torch.save(net.state_dict(), save_path)
                if len(last5loss)<5:
                    last5loss.append(val_loss)
                    # torch.save(net.state_dict(), save_path3)
                else:
                    if last5loss == sorted(last5loss) and not early_stop:
                        torch.save(net.state_dict(), save_path3)
                        early_stop = True
                    else:
                        last5loss = last5loss[0:5] + [val_loss]

        
            print('Finished Training')
            torch.save(net.state_dict(), save_path2)
            if not early_stop:
                torch.save(net.state_dict(), save_path3)
 
if __name__ == '__main__':
    main()