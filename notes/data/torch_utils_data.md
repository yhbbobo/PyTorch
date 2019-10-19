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
## 二、Data Loading Order and Sampler
对于 iterable-style datasets，数据加载顺序完全由用户定义的可迭代样式控制，这样可以更轻松地实现块读取和动态批次大小的实现（例如，通过每次生成一个批次的样本）。本节的其余部分涉及 map-style datasets 的情况。

torch.utils.data.Sampler类用于指定数据加载中使用的索引/键的顺序。它们代表数据集索引上的可迭代对象。例如，在SGD常见情况下，Sampler可以随机排列一列索引，一次生成每个索引，或者为小批量SGD生成少量索引。

基于 DataLoader 的 shuffle 参数，将自动构造顺序或随机排序的采样器。或者，用户可以使用 sampler 参数指定一个自定义Sampler对象，该对象每次都会产生要提取的下一个索引/键。

可以一次生成批量索引列表的自定义采样器作为batch_sampler参数传递。也可以通过batch_size和drop_last参数启用自动批处理。

iterable-style datasets 不能和 sample/ batch_dample 一起使用， 因为iterable-style datasets 没有 index 和 key的概念

## 三、Loading Batched and Non-Batched Data
DataLoader 支持通过参数 batch_size，drop_last 和 batch_sampler 自动将各个提取的数据样本整理为批次

### 1、Automatic batching (default)

这是最常见的情况，对应于获取一小批数据并将其整理为批处理样本，即包含一个张量的张量，其中一维为批处理尺寸（通常是第一维）。

当 batch_size（默认值1）不为None时，数据加载器将生成批处理的样本，而不是单个样本。 batch_size 和 drop_last 参数用于指定数据加载器如何获取数据集keys的批处理。对于map-style datasets，用户可以选择指定batch_sampler，一次生成一个键列表.

batch_size和drop_last参数本质上用于从sampler构造一个 batch_sampler。对于map-style datasets，sampler 可以由用户提供，也可以根据随机参数构造。对于 iterable-style datasets，sampler 是一个 dummy infinite one ，这是什么？

从 iterable-style datasets with multi-processing 中获取数据时，drop_last参数会删除每个worker的数据集副本的最后一个非完整批次。

使用来自采样器的索引获取样本列表后，作为 collat​​e_fn 参数传递的函数用于将样本列表整理为批次。自定义 collat​​e_fn 可用于自定义排序规则，例如将连续数据填充到批处理的最大长度。

### 2、Disable automatic batching

在某些情况下，用户可能希望手动处理批处理，或仅加载单个样本。例如，直接加载批处理的数据（例如，从数据库中批量读取或读取连续的内存块）可能更容易，或者批处理大小取决于数据，或者该程序设计为可处理单个样本。在这种情况下，最好不要使用自动批处理（其中使用collat​​e_fn来整理样本），而应让数据加载器直接返回数据集对象的每个成员。

当 batch_size和batch_sampler均为“None”（batch_sampler的默认值已为“None”）时，将禁用自动批处理。从数据集中获得的每个样本都将作为 collat​​e_fn 参数传递的函数进行处理。禁用自动批处理后，默认的 collat​​e_fn 会将NumPy数组简单地转换为PyTorch张量，并使其他所有内容保持不变。

### 3、Working with collate_fn
禁用自动批处理后，将对每个单独的数据样本调用 collat​​e_fn，并从数据加载迭代器产生输出。在这种情况下，默认的collat​​e_fn会简单地转换PyTorch张量中的NumPy数组。

启用自动批处理后，每次都会使用数据样本列表调用 collat​​e_fn。期望将输入样本整理为一批，以便从数据加载器迭代器中输出。

例如，如果每个数据样本都包含一个3通道图像和一个完整的类标签，即数据集的每个元素都返回一个元组（image，class_index），默认的collat​​e_fn将这样的元组列表整理为批处理图像张量和批处理类标签Tensor的单个元组。特别是，默认的collat​​e_fn具有以下属性：1 它始终为批次维度添加新的维度；2 它会自动将NumPy数组和Python数值转换为PyTorch张量；3 它保留了数据结构，例如，如果每个样本都是一个字典，它将输出一个具有相同键集但将批处理张量作为值的词典（或者如果值不能转换为张量则列出）。

用户可以使用定制的 collat​​e_fn 来实现定制批处理，例如，沿着除第一个维度之外的其他尺寸进行校对，各种长度的填充序列或添加对定制数据类型的支持。
## 四、Single- and Multi-process Data Loading
默认情况下，DataLoader使用单进程数据加载。在Python进程中，全局解释器锁（GIL）防止跨线程真正地完全并行化Python代码。为了避免在加载数据时阻塞计算代码，PyTorch提供了一个简单的开关，只需将参数 num_workers 设置为正整数即可执行多进程数据加载

### 1、Single-process data loading (default)

在此模式下，数据获取是在初始化DataLoader的同一进程中完成的。因此，数据加载可能会阻止计算。然而，当用于在进程之间共享数据的资源（例如，共享存储器，文件描述符）有限时，或者当整个数据集很小并且可以完全加载到存储器中时，该模式可能是优选的。此外，单进程加载通常显示更多可读的错误跟踪，因此对于调试很有用。

### 2、Multi-process data loading

将参数num_workers设置为正整数将打开具有指定数量的加载程序工作进程的多进程数据加载。在这种模式下，每次创建DataLoader的迭代器时（例如，当您调用enumerate（dataloader）时），都会创建 num_workers个工作进程。此时，数据集collate_fn 和 worker_init_fn 被传递给每个工作程序，在这里它们被用来初始化和获取数据。这意味着数据集访问及其内部IO转换（包括collat​​e_fn）在工作进程中运行。

torch.utils.data.get_worker_info() 在工作进程中返回各种有用的信息（包括工作ID，数据集副本，初始种子等），并在主进程中返回None。用户可以在数据集代码或worker_init_fn 中使用此函数来分别配置每个数据集副本，并确定代码是否在工作进程中运行。例如，这在sharding the dataset时特别有用。

对于map-style datasets，主要过程使用采样器生成索引并将其发送给工作人员。因此，任何随机播放都是在主过程中完成的，该过程通过为索引分配索引来引导加载。

对于iterable-style datasets，由于每个工作进程都获得了数据集对象的副本，因此幼稚的多进程加载通常会导致数据重复。使用torch.utils.data.get_worker_info（）或worker_init_fn，用户可以独立配置每个副本。 出于类似的原因，在多进程加载中，drop_last 参数删除每个工作程序的可迭代样式数据集副本的最后一个非完整批次。

一旦迭代结束或迭代器被垃圾回收，Workers 将关闭。

warning:通常不建议在多进程加载中返回CUDA张量，因为在使用CUDA和在多处理中共享CUDA张量时存在许多细微之处（请参阅 CUDA in multiprocessing）。相反，我们建议使用自动内存固定（即设置 pin_memory = True），这样可以将数据快速传输到支持CUDA的GPU。

由于工作程序依赖于Python多处理，因此与Unix相比，Windows上的工作程序启动行为有所不同：

* 在Unix上，fork（）是默认的多处理启动方法。使用fork（），子工作人员通常可以直接通过克隆的地址空间访问数据集和Python参数函数。

* 在Windows上，spawn（）是默认的多处理启动方法。使用spawn（），将启动另一个解释器，该解释器运行您的主脚本，然后运行内部工作程序函数，该函数通过pickle序列化接收数据集，collat​​e_fn和其他参数。

这种独立的序列化意味着应该采取两个步骤来确保在使用多进程数据加载时与Windows兼容：
* 1、用__name__ =='__main__'：将大部分主脚本代码包装起来，以确保启动每个辅助进程时它不会再次运行（很可能会产生错误）。
* 2、确保在__main__检查之外将任何自定义collat​​e_fn，worker_init_fn或数据集代码声明为顶级定义。这样可以确保它们在工作进程中可用。（这是必需的，因为将函数仅作为引用而不是字节码进行腌制。）

多进程数据加载中的随机性：默认情况下，每个工作程序的 PyTorch种子将设置为base_seed + worker_id，其中base_seed是主进程使用其RNG生成的长整数（因此，强制使用RNG状态）。但是，初始化工作程序（例如NumPy）时，可能会复制其他库的种子，导致每个工作程序返回相同的随机数。在 worker_init_fn 中，您可以使用torch.utils.data.get_worker_info（）。seed或torch.initial_seed（）访问每个工作人员的PyTorch种子集，并在加载数据之前使用它为其他库添加种子。
## 五、Memory Pinning
主机到GPU副本源自固定（页面锁定）内存时，速度要快得多。有关一般何时以及如何使用固定内存的更多详细信息，请参见使用固定内存缓冲区

对于数据加载，将pin_memory = True传递给DataLoader将自动将获取的数据张量放入固定内存中，从而能够更快地将数据传输到支持CUDA的GPU。

默认的内存固定逻辑仅识别张量以及包含张量的映射和可迭代对象。默认情况下，如果固定逻辑看到一个属于自定义类型的批处理（如果您具有返回自定义批处理类型的collat​​e_fn就会发生），或者如果该批处理的每个元素都是自定义类型，则固定逻辑将无法识别它们，它将返回该批处理（或那些元素）而无需固定内存。要为自定义批处理或数据类型启用内存固定，请在您的自定义类型上定义pin_memory（）方法：
```python
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```
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
[2] https://zhuanlan.zhihu.com/p/85385325   

