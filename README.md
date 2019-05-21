# pytorch1.0.0学习笔记  
![](imgs/dl.png)

## 一、资源
[官网](https://pytorch.org/) | [GitHub](https://github.com/pytorch/pytorch) | [Examples](https://github.com/pytorch/examples)  | [Tutorials](https://github.com/pytorch/tutorials) | [API](https://pytorch-cn.readthedocs.io/zh/latest/#pytorch)  | [apachecn](https://github.com/apachecn/pytorch-doc-zh)  | [tnt](https://github.com/pytorch/tnt)  

   
## 二、官方Tutorial学习笔记   
[备忘录](https://pytorch.org/tutorials/beginner/ptcheat.html)   
### 1.Getting Started  
* 60分钟闪电战
    * [什么是pytorch？](notes/pytorch.md)
    * [自动微分](notes/autograd.md)
    * [神经网络](notes/nn.md)
    *  
    * [训练分类器](notes/training_classifier.md)
    * [torch.nn](code/nn_tutorial.ipynb)
    * 
    * [Data Parallelism](notes/dataparallelism.md) 
    * 
    * [数据加载和处理教程](notes/load_pre.md)
    * [迁移学习](code/transferlearning.ipynb)  
    * [模型保存与加载](notes/load_save_model.md)


### 2.分布式
#### (1)数据并行
* [Pytorch单机多卡和多机多卡分布式](notes/multigpus.md)

#### (2).模型并行分布式
* [官方demo](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
#### (3).数据并行&模型并行 同时使用
* [官方Demo](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
### 3.API学习
* [分布式训练原理](notes/distributed.md) 
## 三、总结笔记
### 1. 数据预处理
* [ImageFolder使用](notes/imagefolder.md)   
* [DataSet使用](code/dataSet.py)
* [DataLoader使用](code/dataLoader.py)
### 2. 定义模型 

### 3. 训练模型

### 4. 部署/推理

  
   
**---------------------------------------------------------**   
# 参考
[1] [BenchMark](https://github.com/fusimeng/framework_benchmark) 