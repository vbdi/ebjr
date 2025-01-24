'''
Test script for EBJR CIFAR-10/100
Copyright (c) Huawei Technologies Canada Co. Ltd., 2021
'''

from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import numpy as np
import copy
from op_counter import measure_model

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='EBJR CIFAR10/100 Testing')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-tc', '--tcheckpoint', default='teacher checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='checkpoints/cifar10/densenet-bc-52-6/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tresume', default='checkpoints/cifar10/densenet-bc-64-12/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest teacher checkpoint (default: none)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=52, help='Model depth.')
parser.add_argument('--tarch', '-ta', metavar='ARCH', default='densenet',
                    choices=model_names,
                    help='Teacher model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--tdepth', type=int, default=64, help='Teacher model depth.')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')

parser.add_argument('--router_threshold', '--th', default=2.46, type=float,
                    metavar='Threshold', help='Selector Threshold')

parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')

parser.add_argument('--tblock-name', type=str, default='BasicBlock',
                    help='the building block for teacher Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')

parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=6, help='Growth rate for DenseNet.')
parser.add_argument('--tgrowthRate', type=int, default=12, help='Teacher Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

from collections import OrderedDict

def main():
    global best_acc    

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    # Teacher Model
    print("==> creating teacher model '{}'".format(args.tarch))
    if args.tarch.startswith('resnext'):
        teacher_model = models.__dict__[args.tarch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.tdepth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.tarch.startswith('densenet'):
        teacher_model = models.__dict__[args.tarch](
                    num_classes=num_classes,
                    depth=args.tdepth,
                    growthRate=args.tgrowthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.tarch.startswith('wrn'):
        teacher_model = models.__dict__[args.tarch](
                    num_classes=num_classes,
                    depth=args.tdepth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.tarch.endswith('resnet'):
        teacher_model = models.__dict__[args.tarch](
                    num_classes=num_classes,
                    depth=args.tdepth,
                    block_name=args.tblock_name,
                )
    else:
        teacher_model = models.__dict__[args.tarch](num_classes=num_classes)

    model = torch.nn.DataParallel(model)
    teacher_model = torch.nn.DataParallel(teacher_model)
    if use_cuda:
        model.cuda()
        teacher_model.cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('    Total teacher params: %.2fM' % (sum(p.numel() for p in teacher_model.parameters())/1000000.0))

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        if use_cuda:
            checkpoint = torch.load(args.resume)
        else:
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))        
        best_acc = checkpoint['best_acc']

        model.load_state_dict(checkpoint['state_dict'])

        assert os.path.isfile(args.tresume), 'Error: no teacher checkpoint directory found!'
        args.tcheckpoint = os.path.dirname(args.tresume)
        if use_cuda:
            tcheckpoint = torch.load(args.tresume)
        else:
            tcheckpoint = torch.load(args.tresume, map_location=torch.device('cpu'))        
        tbest_acc = tcheckpoint['best_acc']
        teacher_model.load_state_dict(tcheckpoint['state_dict'])


    model_copy = copy.deepcopy(model)
    teacher_model_copy = copy.deepcopy(teacher_model)

    ### RANet code for FLOPS
    student_flops=0   
    print('---------------------')
    print('Student: ')
    cls_ops, cls_params = measure_model(model_copy, 32, 32)
    student_flops = (cls_ops[0]*2)/(math.pow(10,8))
    print('FLOPs (10^8): ' + str(student_flops))

    teacher_flops=0
    print('---------------------')
    print('Teacher: ')
    teacher_cls_ops, teacher_cls_params = measure_model(teacher_model_copy, 32, 32)
    teacher_flops = (teacher_cls_ops[0]*2)/(math.pow(10,8))
    print('FLOPs (10^8): ' + str(teacher_flops))
    ###

    print('\nEvaluation only')
    min_energy = 2.4066114
    max_energy = 2.4611502
    for router_threshold in [args.router_threshold]:#np.arange(min_energy,max_energy+(max_energy-min_energy)/8.0, (max_energy-min_energy)/8.0):
    ### Cifar100 (energy_fixed)
    #min_energy = 4.615475
    #max_energy = 4.622207
    #for router_threshold in np.arange(min_energy,max_energy, (max_energy-min_energy)/5.0):
        batch_idx, exit_rate, acc, mflops, student_flops, teacher_flops, avg_run_time = test(testloader, model, teacher_model, router_threshold, student_flops, teacher_flops,  use_cuda)
        print('---------------------')
        print('#Samples: ' + str(batch_idx))
        #print('Router Threshold: ' + str(router_threshold))
        print('Samples Processed by Student (%): ' + str(exit_rate * 100))
        print('Samples Processed by Teacher (%): ' + str((1.0 - exit_rate) * 100))
        print('Accuracy (%): ' + str(acc))
        print('FLOPs (10^8): ' + str(mflops))
        print('Average Latency (s): ' + str(avg_run_time))

import math
import torch.nn.functional as F
def test(testloader, model, teacher_model, router_threshold, student_flops, teacher_flops, use_cuda):
    teacher_count = 0

    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    teacher_model.eval()    
    
    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    running_time = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            start_time = time.time()
            outputs = model(inputs)
            soft_outputs = F.softmax(outputs, dim=1)

            energy_value = (1.0 * torch.logsumexp(soft_outputs/ 1.0, dim=1))
            NoExit = energy_value < router_threshold

            if NoExit:            
                 outputs = teacher_model(inputs)
                 teacher_count=teacher_count+1

            running_time.append(time.time() - start_time)

            # measure accuracy and record loss
            [prec1, prec5], _, _ = accuracy(outputs.data, targets.data, topk=(1, 5))

            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
            #if batch_idx==100:
            #    break
        bar.finish()

    batch_idx=batch_idx+1
    exit_rate = (batch_idx - teacher_count)/batch_idx
    mflops = ((exit_rate*batch_idx*student_flops)+((1.0-exit_rate)*batch_idx*(student_flops+teacher_flops))) / batch_idx

    return batch_idx, exit_rate, top1.avg.cpu().detach().numpy(), mflops, student_flops, teacher_flops, np.mean(running_time)

if __name__ == '__main__':
    main()

