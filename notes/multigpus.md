# 多卡训练
2019-04-19   
当模型和数据很大时，单个GPU无法短时间训练完模型，所以需要多GPU来训练模型。而多GPU又分为两种，即**单机多卡**和**多机多卡**。  
* 单机多卡使用torch.nn.DataParallel   
    * [参考网址1](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)   
    * [参考网址2](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)
* 多机多卡使用torch.distributed
    * 
  
## 二、单机多卡
