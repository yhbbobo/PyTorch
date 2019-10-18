# torch.utils.data功能理解   
[源码链接](https://github.com/pytorch/pytorch/tree/master/torch/utils/data)   
torch.utils.data.DataLoader类是PyTorch数据加载实用程序的核心。它是个可在数据集上迭代的Python迭代器，并支持   
* map-style and iterable-style datasets,
* customizing data loading order,
* automatic batching,
* single- and multi-process data loading,
* automatic memory pinning.

DataLoader构造函数的参数配置 ：
```
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
```
以下各节详细介绍了这些选项的效果和用法。
## 一、数据集类型
DataLoader 构造函数最重要的参数是dataset，它指示要从中加载数据的数据集对象。PyTorch支持两种不同类型的数据集：     
* map-style datasets,
* iterable-style datasets.  
### 1、map-style datasets
torch.utils.data.Dataset    

map-style的数据集是一种实现__getitem__()和__len__()协议的数据集。它表示从索引/关键字（可能是非整数）到数据样本的映射。   

使用进行访问时，这样的数据集dataset[idx]可以从磁盘上的文件夹中读取第idx个图像及其对应的标签。   

torch.utils.data.Dataset代表的抽象类Dataset，代表从键到数据样本的映射的所有数据集都应将其子类化。所有子类都应该覆盖__getitem__()，以支持获取给定键的数据样本。子类也可以选择覆盖__len__()，这可以通过许多Sampler实现和默认选项来返回数据集的大小DataLoader。   

### 2、Iterable-style datasets
可迭代样式的数据集是IterableDataset 实现__iter__()协议的子类的实例，并且表示数据样本上的可迭代。这种类型的数据集特别适用于以下情况：随机读取费劲甚至不大可能，并且批处理大小取决于所获取的数据。

例如，这样的数据集调用iter(dataset)，可以返回从数据库，远程服务器甚至实时生成的日志中读取的数据流。
For example, such a dataset, when called iter(dataset), could return a stream of data reading from a database, a remote server, or even logs generated in real time.

iterable-style dataset 是IterableDataset 的子类， 实现了__iter__和 __add__协议 ，表示数据样本上的可迭代对象。这种类型的dataset特别适用于以下情况：随机读取代价高且批处理大小取决于所获取的数据。IterableDataset 见详细细节

将 IterableDataset用于多进程数据加载（multi-process data loading）时，在每个工作进程上都复制相同的数据集对象，因此必须对副本进行不同的配置，以避免重复的数据，有关如何实现此目的，请参见 IterableDataset文档。

# torch.utils.data包理解
## 一、 Dataset
dataset.py   
### 1、class torch.utils.data.Dataset  
表示Dataset的抽象类。所有其他数据集都应该进行子类化。 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)。   
```python
class Dataset(object):
	# 强制所有的子类override getitem和len两个函数，否则就抛出错误；
	# 输入数据索引，输出为索引指向的数据以及标签；
	def __getitem__(self, index):
		raise NotImplementedError
	
	# 输出数据的长度
	def __len__(self):
		raise NotImplementedError
		
	def __add__(self, other):
		return ConcatDataset([self, other])
```
### 2、class torch.utils.data.TensorDataset(*tensors)
Dataset的子类。包装tensors数据集；输入输出都是元组； 通过沿着第一个维度索引一个张量来回复每个样本。 个人感觉比较适用于数字类型的数据集，比如线性回归等。
```python
class TensorDataset(Dataset):
	def __init__(self, *tensor):
		assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
		self.tensors = tensors
		
	def __getitem__(self, index):
		return tuple(tensor[index] for tensor in tensors
		
	def __len__(self):
		return self.tensors[0].size(0)
```
### 3、class torch.utils.data.ConcatDateset(datasets)
连接多个数据集。 目的：组合不同的数据集，可能是大规模数据集，因为连续操作是随意连接的。 datasets的参数：要连接的数据集列表 datasets的样式：iterable
```python
class ConcatDataset(Dataset):
	@staticmethod
	def cumsum(sequence):
		# sequence是一个列表，e.g. [[1,2,3], [a,b], [4,h]]
		# return 一个数据大小列表，[3, 5, 7], 明显看的出来包含数据多少，第一个代表第一个数据的大小，第二个代表第一个+第二数据的大小，最后代表所有的数据大学；
	...
	def __getitem__(self, idx):
		# 主要是这个函数，通过bisect的类实现了任意索引数据的输出；
		dataset_idx = bisect.bisect_right(self.cumulative_size, idx)
		if dataset_idx == 0:
			sample_idx == idx
		else:
			sample_idx = idx - self.cumulative_sizes[dataset_idx -1]
		return self.datasets[dataset_idx][sample_idx]
	...
```
### 4、class torch.utils.data.Subset(dataset, indices)
选取特殊索引下的数据子集； dataset：数据集； indices：想要选取的数据的索引；
### 5、class torch.utils.data.random_split(dataset, lengths):
随机不重复分割数据集； dataset：要被分割的数据集 lengths：长度列表，e.g. [7, 3]， 保证7+3=len(dataset)
## 二、 DataLoader
dataloader.py   
### 1、class torch.utils.data.DataLoader
```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)   
```

数据加载器。 组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。 参数：
* dataset (Dataset) - 从中加载数据的数据集。
* batch_size (int, optional) - 批训练的数据个数。
* shuffle (bool, optional) - 是否打乱数据集（一般打乱较好）。
* sampler (Sampler, optional) - 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
* batch_sampler (Sample, optional) - 和sampler类似，返回批中的索引。
* num_workers (int, optional) - 用于数据加载的子进程数。
* collate_fn (callable, optional) - 合并样本列表以形成小批量。
* pin_memory (bool, optional) - 如果为True，数据加载器在返回去将张量复制到CUDA固定内存中。
* drop_last (bool, optional) - 如果数据集大小不能被batch_size整除， 设置为True可以删除最后一个不完整的批处理。
* timeout (numeric, optional) - 正数，收集数据的超时值。
* worker_init_fn (callabel, optional) - If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None)
特别重要：DataLoader中是不断调用DataLoaderIter
### 2、class _DataLoaderIter(loader)
从DataLoader’s数据中迭代一次。其上面DataLoader功能都在这里DataLoaderIter
```python
class _DataLoaderIter(loader)
```
从DataLoader’s数据中迭代一次。其上面DataLoader功能都在这里；
## 三、 Sampler
sampler.py   
### 1、class torch.utils.data.sampler.Sampler(data_source)
所有采样器的基础类； 每个采样器子类必须提供一个__iter__方法，提供一种迭代数据集元素的索引的方法，以及返回迭代器长度__len__方法。
```python
class Sampler(object):
	def __init__(self, data_source):
		pass
		
	def __iter__(self):
		raise NotImplementedError
		
	def __len__(self):
		raise NotImplementedError
```
### 2、class torch.utils.data.SequentialSampler(data_source)
样本元素顺序排列，始终以相同的顺序。 参数：-data_source (Dataset) - 采样的数据
### 3、class torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None)
样本随机排列，如果没有Replacement，将会从打乱的数据采样，否则，。。 参数：

* data_source (Dataset) - 采样数据
* num_samples (int) - 采样数据大小，默认是全部。
* replacement (bool) - 是否放回

### 4、class torch.utils.data.SubsetRandomSampler(indices)
从给出的索引中随机采样，without replacement。 参数：
* indices (sequence) - 索引序列。
### 5、class torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
将采样封装到批处理索引。 参数：
* sampler (sampler) - 基本采样
* batch_size (int) - 批大小
* drop_last (bool) - 是否删掉最后的批次
### 6、class torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)
样本元素来自[0,…,len(weights)-1]， 给定概率（权重）。 参数：
* weights (list) - 权重列表。不需要加起来为1
* num_samplers (int) - 要采样数目
* replacement (bool) -
## 四、 Distributed
distributed.py
### 1、class torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None)

没读呢


# 使用实例
## 一、Dataset使用
### 1、init
具有一下图像数据如下表示：
```
train
    normal
        1.png
        2.png
        …
        8000.png
    tumor
        1.png   
        2.png
        …
        8000.png
validation
    normal
        1.png
    tumor
        1.png
```
希望能够训练模型，使得能够识别tumor, normal两类，将tumor–>1, normal–>0。
### 2 数据读取
在PyTorch中数据的读取借口需要经过，Dataset和DatasetLoader (DatasetloaderIter)。下面就此分别介绍。

#### Dataset
首先导入必要的包。
```python
import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)
```
其次定义MyDataset类，为了代码整洁精简，将不必要的操作全删，e.g. 图像剪切等。
```python
class MyDataset(Dataset):
	
	def __init__(self, root, size=229, ):
		"""
		Initialize the data producer
		"""
		self._root = root
		self._size = size
		self._num_image = len(os.listdir(root))
		self._img_name = os.listdir(root)
	
	def __len__(self):
		return self._num_image
		
	def __getitem__(self, index):
		img = Image.open(os.path.join(self._root, self._img_name[index]))
		
		# PIF image: H × W × C
		# torch image: C × H × W
		img = np.array(img, dtype-np.float32).transpose((2, 0, 1))
		
		return img
```
#### DataLoader
将MyDataset封装到loader器中。
```python
from torch.utils.data import DataLoader

# 实例化MyData
dataset_tumor_train = MyDataset(root=/img/train/tumor/)
dataset_normal_train = MyDataset(root=/img/train/normal/)
dataset_tumor_validation = MyDataset(root=/img/validation/tumor/)
dataset_normal_validation = MyDataset(root=/img/validation/normal/)

# 封装到loader
dataloader_tumor_train = DataLoader(dataset_tumor_train, batch_size=10)
dataloader_normal_train = DataLoader(dataset_normal_train, batch_size=10)
dataloader_tumor_validation = DataLoader(dataset_tumor_validation, batch_size=10)
dataloader_normal_validation = DataLoader(dataset_normal_validation, batch_size=10)
```
### 3 train_epoch
简单将数据流接口与训练连接起来
```python
def train_epoch(model, loss_fn, optimizer, dataloader_tumor, dataloader_normal):
	model.train()
	
	# 由于tumor图像和normal图像一样多，所以将tumor，normal连接起来，steps=len(tumor_loader)=len(normal_loader)
	steps = len(dataloader_tumor)
	batch_size = dataloader_tumor.batch_size
	dataiter_tumor = iter(dataloader_tumor)
	dataiter_normal = iter(dataloader_normal)
	
	for step in range(steps):
		data_tumor = next(dataiter_tumor)
		target_tumor = [1, 1,..,1] # 和data_tumor长度相同的tensor
		data_tumor = Variable(data_tumor.cuda(async=True))
		target_tumor = Variable(target_tumor.cuda(async=True))
		 
		data_normal = next(dataiter_normal)
		target_normal = [0, 0,..,0] # 
		data_normal = Variable(data_normal.cuda(async=True))
		target_normal = Variable(target_normal.cuda(async=True))
		
		idx_rand = Variable(torch.randperm(batch_size*2).cuda(async=True))
		
		data = torch.cat([data_tumor, data_normal])[idx_rand]
		target = torch.cat([target_tumor, target_normal])[idx_rand]
		output = model(data)
		loss = loss_fn(output, target)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		probs = output.sigmoid()
```
## 二、github
https://github.com/fusimeng/CommonTool   
https://github.com/LianHaiMiao/pytorch-lesson-zh/   

# Reference
[1] https://likewind.top/2019/02/01/Pytorch-dataprocess/   
