# pytorch1.3.0学习笔记  
![](imgs/dl.png)
## 一、资源
[官网](https://pytorch.org/) | [GitHub](https://github.com/pytorch/pytorch) | [Examples](https://github.com/pytorch/examples)  | [Tutorials](https://github.com/pytorch/tutorials) | [API](https://pytorch-cn.readthedocs.io/zh/latest/#pytorch)  | [apachecn](https://github.com/apachecn/pytorch-doc-zh)  | [tnt](https://github.com/pytorch/tnt)  | [d2l](https://github.com/ShusenTang/Dive-into-DL-PyTorch) 
## 二、[基础&学习笔记](notes/pytorch-api.md)
[什么是pytorch？](notes/pytorch.md)   
[备忘录](https://pytorch.org/tutorials/beginner/ptcheat.html)  
### 0、数据处理
* [数据加载和处理教程](notes/data/load_pre.md)
* [torch.utils.data](notes/data/torch_utils_data.md)
* [torchvision.datasets](notes/data/torch_torchvision.md)
* [ImageFolder使用](notes/imagefolder.md)   
* [DataSet使用](code/dataSet.py)
* [DataLoader使用](code/dataLoader.py)

### 1、网络模型
* [torch.nn package](notes/nn/torch_nn.md)
* [神经网络](notes/nn/nn.md)
* [自动微分](notes/nn/autograd.md)
* [训练分类器](notes/nn/training_classifier.md)
* [模型加载/参数初始化/Finetune](notes/lif.md)
* [torch.nn](code/nn_tutorial.ipynb)
### 2、损失函数

### 3、BP

### 4、优化器
   
### 5、模型保存&迁移学习&...
* 模型保存
    * [模型保存与加载](notes/load_save_model.md)
* 迁移学习
    * [迁移学习](code/transferlearning.ipynb)
### 6、分布式
[分布式训练原理](notes/distributed.md)    
[torch.distributed](notes/torch-distributed.md)
* 模型并行
    * [官方demo](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
* 数据并行
    * [Pytorch单机多卡和多机多卡分布式](notes/multigpus.md)
    * [Data Parallelism](notes/dataparallelism.md) 
* 模型并行&数据并行
    * [官方Demo](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
### 6、加速运行
* [自定义 C++ 与 CUDA 拓展](notes/cuda.md)
  
## 三、源码学习
[torch包的目录结构](notes/api/torch_arch.md)
* [torch](notes/api/torch.md)
* [torch.nn](notes/api/torch_nn.md)
* [torch.nn.functional](notes/api/torch_nn_funtional.md)
* [torch.Tensor](notes/api/torch_tensor.md)
* [torch.Attributes](notes/api/torch_attributes.md)
* [torch.autograd](notes/api/torch_autograd.md)
* [torch.cuda](notes/api/torch_cuda.md)
torch.distributed
torch.distributions
torch.hub
torch.jit
torch.nn.init
torch.onnx
torch.optim
Quantization
torch.random
torch.sparse
torch.Storage
torch.utils.bottleneck
torch.utils.checkpoint
torch.utils.cpp_extension
torch.utils.data
torch.utils.dlpack
torch.utils.model_zoo
torch.utils.tensorboard
Type Info
Named Tensors
Named Tensors operator coverage
torch.__config__

# 参考
[1] [BenchMark](https://github.com/fusimeng/framework_benchmark)    
[2]  https://blog.csdn.net/rogerfang/article/details/82291464   
