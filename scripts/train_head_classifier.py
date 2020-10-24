import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optparse import OptionParser
import json
import sys
import os

import torchvision.transforms as transforms
sys.path.insert(0,'../model/')
sys.path.insert(0,'../utils/')

from CifarDataset import CIFAR100
from resnet import resnet18


import logging
import datetime


def get_args():

    parser = OptionParser()

    parser.add_option('--batch_size', dest='batch_size', default=12,
                      type='int', help='batch size')

    parser.add_option('--lr', dest='lr', default=0.01,
                      type='float', help='learning rate')
    
    parser.add_option('--use_cuda', dest='use_cuda', type='str',default='True', help='use cuda')

    parser.add_option('--checkpoint_file', dest='checkpoint_file',type='str',
                      default=None, help='load model file')

    parser.add_option('--source_model', dest='source_model',type='str',
                      default=None, help='load source model')

    parser.add_option('--gamma',dest = 'gamma',type = 'float', default = 0.2,help = 'lr decay')

    parser.add_option('--step_size',dest = 'step_size',type = 'int',default = 100,help = 'step_size')

    parser.add_option('--max_epoch',dest = 'max_epoch',default = 100,type = 'int',help = 'max_iter')

    parser.add_option('--snapshot',dest = 'snapshot',default = 100,type = 'int',help = 'snapshot')

    parser.add_option('--dataset', dest = 'dataset', default = 'CIFAR', help = 'CIFAR')   

    parser.add_option('--net', dest = 'net', default = 'resnet18', help = 'net type')

    parser.add_option('--gpu_id',dest = 'gpu_id',default = None, type = 'int',help = 'gpu id')

    parser.add_option('--optimizer',dest = 'optimizer',default = 'SGD', type = 'str',help = 'optimizer')

    parser.add_option('--train_head_classifier',dest = 'train_head_classifier',default = 'true', type = 'str',help = 'train_head_classfier')

    parser.add_option('--num_category',dest = 'num_category',default = 100, type = 'int' ,help = 'num_category of target dataset')

    (options, args) = parser.parse_args()
    return options


def str2bool(s):

    if s == 'False' or s == 'false':
        return False

    elif s == 'True' or s == 'true':
        return True

    else:
        raise Exception(print('s should be string: True, true or false, False'))


def main():

    args = get_args()

    use_cuda = str2bool(args.use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)

    batch_size = args.batch_size

    criterion = nn.NLLLoss()

    learning_rate = args.lr

    num_epochs = args.max_epoch

 
    checkpoint_save_dir = os.path.join('../checkpoints/%s_%s_finetuned_from_imagenet'%(args.net,args.dataset))
    if not os.path.exists(checkpoint_save_dir):
      os.makedirs(checkpoint_save_dir)


    if args.dataset == 'CIFAR':
        composed_transform = transforms.Compose([transforms.Resize((224,224),3),transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        train_set = CIFAR100(root='../data/CIFAR100/', download=True, train = True, transform=composed_transform)
        test_set = CIFAR100(root='../data/CIFAR100/', download=True, train = False,transform=composed_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers = 4)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers = 4)



    if args.net == 'resnet18':

        #specify train head classifier
        # the default num_classes of imagenet is 1000
        net = resnet18(num_classes=args.num_category, train_head_classfier=str2bool(args.train_head_classifier)).to(device)

        if args.source_model:

            # load feature extractor weights
            pretrained_dict = torch.load(args.source_model,map_location='cuda:%d'%args.gpu_id)
            net_dict = net.state_dict()
            feature_extractor = {k:v for k,v in pretrained_dict.items() if k[0:2] != 'fc'}
            net_dict.update(feature_extractor)
            net.load_state_dict(net_dict)
            print ('load source model successfully', args.source_model)

        

        if args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

        for epoch in range(num_epochs):
            net.train()
            total = 0.0
            correct = 0.0
            for i, batch_train in enumerate(train_loader): 
                # Move tensors to the configured device
                images = batch_train[0].to(device)
                labels = batch_train[1].to(device)
                
                outputs = F.softmax(net(images))
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            train_acc = 100 * correct / total
            with torch.no_grad():
                net.eval()
                correct = 0
                total = 0
                for batch_test in test_loader:
                    images = batch_test[0].to(device)
                    labels = batch_test[1].to(device)
                    outputs = F.softmax(net(images))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                test_acc = 100 * correct / total
                print('Epoch: %d, lr: %.6f, train_acc: %.2f, test_acc: %.2f'%(epoch, learning_rate,train_acc, test_acc))


            # learning rate decay
            if (epoch + 1) % args.step_size == 0:
                learning_rate = learning_rate * args.gamma
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    

            # save intermediate checkpoints
            if (epoch + 1) % args.snapshot == 0:
                torch.save(net.state_dict(), os.path.join(checkpoint_save_dir,'CP%d_ACC%.2f.pth'%(epoch+1,test_acc)))
                print('Checkpoint saved !')


if __name__ == '__main__':
    main()
