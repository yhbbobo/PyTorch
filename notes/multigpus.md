# 多卡训练
2019-04-19   
* 单机多卡使用torch.nn.DataParallel   
    * [官方DataParallel教程-1](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)   
    * [官方DataParallel教程-2](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)
* 多机多卡使用torch.distributed
    * [官方distributed教程](https://pytorch.org/docs/master/distributed.html)
    * [官方pytorch分布式原理教程](https://github.com/pytorch/tutorials/blob/master/intermediate_source/dist_tuto.rst)
## 一、Pytorch分布式介绍
Pytorch 是从Facebook孵化出来的，在0.4的最新版本加入了分布式模式，比较吃惊的是它居然没有采用类似于TF和MxNet的PS-Worker架构。而是采用一个还在Facebook孵化当中的一个叫做gloo的家伙。   

这里引入了一个新的函数`model = torch.nn.parallel.DistributedDataParallel(model)`为的就是支持分布式模式.   

不同于原来在`multiprocessing`中的`model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()`函数，这个函数只是实现了在单机上的多GPU训练，**根据官方文档的说法，甚至在单机多卡的模式下，新函数表现也会优于这个旧函数**。   
   
这里要提到两个问题：
* 每个进程都有自己的Optimizer同时每个迭代中都进行完整的优化步骤，虽然这可能看起来是多余的，但由于梯度已经聚集在一起并跨进程平均，因此对于每个进程都是相同的，这意味着不需要参数广播步骤，从而减少了在节点之间传输张量tensor所花费的时间。
* 另外一个问题是Python解释器的，每个进程都包含一个独立的Python解释器，消除了来自单个Python进程中的多个执行线程，模型副本或GPU的额外解释器开销和“GIL-thrashing”。 这对于大量使用Python运行时的模型尤其重要。
       

