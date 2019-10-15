# 分布式通信包 - torch.distributed
* V1.2    

* 参考[🔗](https://pytorch.apachecn.org/docs/1.2/distributed.html#)

## 一、后端
![](../imgs/api/01.png)    
* MPI 基础概念：[🔗](https://github.com/fusimeng/ParallelComputing/blob/master/notes/mpiconcept.md)   
* MPI通信[🔗](https://github.com/fusimeng/ParallelComputing/blob/master/notes/communication.md)/[🔗](https://github.com/fusimeng/ParallelComputing/blob/master/notes/CollectiveCommunication.md)   

## 二、PyTorch附带的后端
后端选择原则
## 三、常见的环境变量

## 四、基本
对比
* torch.nn.parallel.DistributedDataParallel()
* torch.multiprocessing 
* torch.nn.DataParallel() 

## 五、初始化
torch.distributed.init_process_group(backend, init_method='env://', timeout=datetime.timedelta(seconds=1800), **kwargs)

参数:

* backend (str or Backend) – 后端使用。根据构建时配置，有效值包括 mpi，gloo和nccl。该字段应该以小写字符串形式给出(例如"gloo")，也可以通过Backend访问属性(例如Backend.GLOO)。
* init_method (str, optional) – 指定如何初始化进程组的URL。
* world_size (int, optional) – 参与作业的进程数。
* rank (int, optional) – 当前流程的排名。
* timeout (timedelta__, optional) – 针对进程组执行的操作超时，默认值等于30分钟，这仅适用于gloo后端。
* group_name (str, optional__, deprecated) – 团队名字。

目前支持三种初始化方法：

### 1、TCP初始化
### 2、共享文件系统初始化
### 3、环境变量初始化



