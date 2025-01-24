# EBJR: Energy-Based Joint Reasoning for Adaptive Inference (BMVC2021)
##### This notebook contains the implmenetation of the paper ['EBJR: Energy-Based Joint Reasoning for Adaptive Inference'](https://www.bmvc2021-virtualconference.com/assets/papers/0502.pdf) published in BMVC2021.

![](https://user-images.githubusercontent.com/38634796/143724800-915267db-4472-4a7f-90f6-a54be489ced1.png)

## Introduction
##### - Smaller (shallower) models -> lower prediction accuracy, but faster
##### - Larger (deeper) models -> higher prediction accuracy, but slower
##### - Combine Small (called Student) + Large (called Teacher)
##### - Majority processed by Student -> high accuracy and/or low latency
##### - Joint inference via an energy-based Router: to decide which model to use for each input data
##### - Inference-time trade-off between latency and accuracy
##### - Applicable to any pre-trained model, and different computer vision tasks

## Requirements
##### In order to run this code, torch==1.7.1 and torchvision==0.8.2 packages are required, which can be installed using the following commands:


```python
!pip install torch==1.7.1
!pip install torchvision==0.8.2
```

## Usage

#### Download Code


```python
!wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ebjr/code.zip
!unzip -qo code.zip
```

##### EBJR over CIFAR-10 with DenseNet-52-6 (as Student) and DenseNet-64-12 (as Teacher):


```python
!python ebjr.py --evaluate --dataset cifar10 --arch densenet --depth 52 --growthRate 6 --tarch densenet --tdepth 64 --tgrowthRate 12 --resume checkpoints/cifar10/densenet-bc-52-6/model_best.pth.tar --tresume checkpoints/cifar10/densenet-bc-64-12/model_best.pth.tar --router_threshold 2.46
```

    ==> Preparing dataset cifar10
    ==> creating model 'densenet'
    ==> creating teacher model 'densenet'
        Total params: 0.07M
        Total teacher params: 0.37M
    ==> Resuming from checkpoint..
    ---------------------
    Student: 
    Params: 0.07M
    FLOPs (10^8): 0.54754784
    ---------------------
    Teacher: 
    Params: 0.37M
    FLOPs (10^8): 2.92380404
    
    Evaluation only
    [KProcessing |################################| (10000/10000) Data: 0.001s | Batch: 0.012s | Total: 0:02:02 | ETA: 0:00:01 | top1:  94.7700 | top5:  99.8000
    ---------------------
    #Samples: 10000
    Samples Processed by Student (%): 72.03
    Samples Processed by Teacher (%): 27.969999999999995
    Accuracy (%): 94.77
    FLOPs (10^8): 1.3653358299879998
    Average Latency (s): 0.010832044768333436
    [?25h

##### EBJR over CIFAR-100 with DenseNet-58-6 (as Student) and DenseNet-88-8 (as Teacher):


```python
!python ebjr.py --evaluate --dataset cifar100 --arch densenet --depth 58 --growthRate 6 --tarch densenet --tdepth 88 --tgrowthRate 8 --resume checkpoints/cifar100/densenet-bc-58-6/model_best.pth.tar --tresume checkpoints/cifar100/densenet-bc-88-8/model_best.pth.tar --router_threshold 4.6195
```

    ==> Preparing dataset cifar100
    ==> creating model 'densenet'
    ==> creating teacher model 'densenet'
        Total params: 0.09M
        Total teacher params: 0.30M
    ==> Resuming from checkpoint..
    ---------------------
    Student: 
    Params: 0.09M
    FLOPs (10^8): 0.64281872
    ---------------------
    Teacher: 
    Params: 0.30M
    FLOPs (10^8): 2.14001928
    
    Evaluation only
    [KProcessing |################################| (10000/10000) Data: 0.001s | Batch: 0.016s | Total: 0:02:40 | ETA: 0:00:01 | top1:  74.7900 | top5:  93.6900
    ---------------------
    #Samples: 10000
    Samples Processed by Student (%): 56.730000000000004
    Samples Processed by Teacher (%): 43.269999999999996
    Accuracy (%): 74.79
    FLOPs (10^8): 1.568805062456
    Average Latency (s): 0.01464376654624939
    [?25h

## Arguments
##### - 'dataset': cifar10, cifar100, or imagenet
##### - 'arch': the network architecture for Student
##### - 'depth': the model depth for Student
##### - 'growthRate': the growth rate for for Student (if tarch is resnet or densenet)
##### - 'tarch': the network architecture for Teacher
##### - 'tdepth': the model depth for Teacher (if tarch is resnet or densenet)
##### - 'tgrowthRate': the growth rate for for Teacher (if arch is densenet)
##### - 'resume': the checkpoint path for Student
##### - 'tresume': the checkpoint path for Teacher
##### - 'router_threshold': the router threshold

## Results

### Image Classification
##### - Average of ~2X less FLOPs compared to Teacher
##### - Average of ~1.5X less latency compared to Teacher

![](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ebjr/table.png)

#### Trade-off curves compared with SOTA

![](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ebjr/ic-curves.png)

### Object Detection
##### With EfficientDet-D0 (as Student) and EfficientDet-D4 (as Teacher) 

![](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ebjr/od.png)

## Specialized EBJR
#####  - Assumption: majority of data (e.g., 70%) belongs to a small subset of popular classes 
#####  - Train and specialize the Student model targeting specific/popular classes
#####  - Highly accurate Student predictions for the majority of input data

## Demo video for Specialized EBJR 
##### The following video demonstrates the Specialized EBJR with Top-50+1 scenario over OID dataset.


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ebjr/Demo.mp4" type="video/mp4">
</video>
```

<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ebjr/Demo.mp4" type="video/mp4">
</video>
