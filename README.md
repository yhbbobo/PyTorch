# pytorch1.0.0学习笔记  
2019-04-18   
## 一、资源
官网：https://pytorch.org/     
GitHub：https://github.com/pytorch/pytorch   
examples:https://github.com/pytorch/examples   
tutorials:https://github.com/pytorch/tutorials    
中文Tutorial：https://github.com/apachecn/pytorch-doc-zh | https://github.com/fusimeng/pytorch-doc-zh       
tnt:https://github.com/pytorch/tnt  

   
   
API：https://github.com/zergtant/pytorch-handbook   
API：https://pytorch.apachecn.org/#/     
API：https://pytorch-cn.readthedocs.io/zh/latest/#pytorch 
## 二、官方Tutorial学习笔记   
[备忘录](https://pytorch.org/tutorials/beginner/ptcheat.html)   
### 1.Getting Started  
* 60分钟闪电战
    * [什么是pytorch？](notes/pytorch.md)
    * [自动微分](notes/autograd.md)
    * [神经网络](notes/nn.md)
    * [训练分类器](notes/training_classifier.md)
    * [Data Parallelism](notes/dataparallelism.md)
* [数据加载和处理教程](notes/load_pre.md)
* [迁移学习](https://github.com/fusimeng/pytorchexamples/blob/master/transferlearning.ipynb)
* [单机多卡和多机多卡分布式](notes/multigpus.md)
    * [单机多卡实现](https://github.com/fusimeng/pytorchexamples/blob/master/single_multigpus.ipynb)
    * [多机多卡实现]()
* [分布式训练原理](notes/distributed.md)
    * [分布式MPI几个概念](https://github.com/fusimeng/ParallelComputing)
    * [官方分布式API](https://pytorch.org/docs/master/distributed.html)  
    * [官方分布式tutorials](https://github.com/pytorch/tutorials/blob/master/intermediate_source/dist_tuto.rst) | [或者](https://pytorch.org/tutorials/intermediate/dist_tuto.html) 
    * [官方ImageNet](https://github.com/pytorch/examples/tree/master/imagenet)  
    * [参考3-博客](https://blog.csdn.net/m0_38008956/article/details/86559432)   
    * [参考4-mnist](https://blog.csdn.net/jacke121/article/details/80605421) 
    * [代码参考](https://github.com/seba-1511/dist_tuto.pth/)    
    

* [ImageFolder使用](https://blog.csdn.net/TH_NUM/article/details/80877435)   
