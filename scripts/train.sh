#!/bin/bash 

dataset="CIFAR"
net="resnet18"
train_head_classifier="true"
max_epoch=100
lr=0.01
batch_size=24
source_model="../pretrained_model/resnet18-5c106cde.pth"
use_cuda="true"
num_category=100


python -u train_head_classifier.py \
--use_cuda $use_cuda \
--dataset $dataset \
--net $net \
--max_epoch $max_epoch \
--lr $lr \
--batch_size $batch_size \
--train_head_classifier $train_head_classifier \
--source_model $source_model \
--num_category $num_category \
