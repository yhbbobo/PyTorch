# GPU 显存不足怎么办？
## 一、前言
最近跑的模型都比较大，尤其是Bert， 这真的是难为我 1080ti 了， 在Bert的Example中，官方提供了一些 Trick 来帮助我们加速训练，很良心， 但感觉还不够，于是花费一些时间整理出一个 Trick 集合，来帮助我们在显存不足的时候来嘿嘿嘿。   
本文分为两大部分，第一部分引入一个主题：如何估计模型所需显存， 第二个主题：GPU显存不足时的各种 Trick 。    
## 二、监控 GPU
监控GPU最常用的当然是nvidia-smi，但有一个工具能够更好的展示信息：gpustat 。    
```
nvidia-smi
watch --color -n1 gpustat -cpu   # 动态事实监控GPU  
```
推荐在配置文件中配置别名，反正我每次 gpu 一下，信息就全出来了，很方便   
## 三、如何估计模型显存   
首先，思考一个问题： 模型中的哪些东西占据了我的显存，咋就动不动就 out of memory？

其实一个模型所占用的显存主要包含两部分： 模型自身的参数， 优化器参数， 模型每层的输入输出。     
### 1、模型自身参数
模型自身的参数指的就是各个网络层的 Weight 和Bias，这部分显存在模型加载完成之后就会被占用， 注意到的是，有些层是有参数的，如CNN， RNN； 而有些层是无参数的， 如激活层， 池化层等。    

从Pytorch 的角度来说，当你执行 model.to(device) 是， 你的模型就加载完毕，此时你的模型就已经加载完成了。

对于Pytorch来说，模型参数存储在 model.parameters() 中，因此，我们不需要自己计算，完全可以通过Pytorh来直接打印：   
```python
print('Model {} : params: {:4f}M'.format(model._get_name()， para * type_size / 1000 / 1000))
```
### 2、优化器参数
优化器参数指的是模型在优化过程即反向传播中所产生的参数， 这部分参数主要指的就是 dw， 即梯度，在SGD中， 其大小与参数一样， 因此在优化期间， 模型的参数所占用的显存会翻倍。

值得注意的是，不同的优化器其所需保存的优化参数不同， 对于 Adam， 由于其还需要保存其余参数， 模型的参数量会在优化区间翻 4 倍。
### 3、模型每层的输入输出
首先，第一点是输入数据所占用的显存， 这部分所占用的显存其实并不大，这是因为我们往往采用迭代器的方式读取数据，这意味着我们其实并不是一次性的将所有数据读入显存(具体的机制并不清楚)，而这保证每次读取的数据与整个网络参数来比是微不足道的。

然后，在模型进行前向传播与反向传播时， 一个很重要的事情就是保存每一层的输出以及其对应的梯度， 这意味着，这也占据了很大一部分显存， 模型输出的显存占用可以总结为：   

* 每一层的输出(多维数组)， 其对应的梯度， 值得注意的是，模型输出不需要存储相应的动量信息（即此处如果使用Adam， 模型输出的参数量依旧是2倍而不是4倍， 我也不知道为啥，，，）
* 输出的显存占用与 batch size 成正比

那么有没有办法通过Pytorch来计算这部分参数量呢？ 答案是有的，我们可以假设一个batch的样本，然后通过 model.modules() 来对每一层进行遍历，获得每一层的输出shape， 然后就能够获得一个batch的数据的输出参数量。[2]   
### 4、所有的显存占用计算
`
显存占用 = 模型自身参数 × n + batch size × 输出参数量 × 2 + 一个batch的输入数据（往往忽略）
`

其中，n是根据优化算法来定的，如果选用SGD， 则 n = 2， 如果选择Adam， 则 n = 4.

一个很棒的实现如下， 我懒得再重新写了，你可以根据这个改一改，问题不大。
```python
# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32 
def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)
    mods = list(model.modules())
    out_sizes = []
    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out
    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))
```
## 四、GPU 显存不足时的Trick [2]
此处不讨论多GPU， 分布式计算等情况，只讨论一些常规的 Trick， 会不定时进行更新。
### 1、降低batch size
这应该很好理解，适当降低batch size， 则模型每层的输入输出就会成线性减少， 效果相当明显。
### 2、选择更小的数据类型
一般默认情况下， 整个网络中采用的是32位的浮点数，如果切换到 16位的浮点数，其显存占用量将接近呈倍数递减。
### 3、精简模型
在设计模型时，适当的精简模型，如原来两层的LSTM转为一层； 原来使用LSTM， 现在使用GRU； 减少卷积核数量； 尽量少的使用 Linear 等。
### 4、数据角度
对于文本数据来说，长序列所带来的参数量是呈线性增加的， 适当的缩小序列长度可以极大的降低参数量。
### 5、total_loss
考虑到 loss 本身是一个包含梯度信息的 tensor， 因此，正确的求损失和的方式为：    
`total_loss += loss.item()`   
### 6、释放不需要的张量和变量
采用del释放你不再需要的张量和变量，这也要求我们在写模型的时候注意变量的使用，不能随心所欲。
### 7、Relu 的 inplace 参数
激活函数 Relu() 有一个默认参数 inplace ，默认为Flase， 当设置为True的时候，我们在通过relu() 计算得到的新值不会占用新的空间而是直接覆盖原来的值，这表示设为True， 可以节省一部分显存。
### 8、梯度累积
首先， 要了解一些Pytorch的基本知识：

* 在Pytorch 中，当我们执行 loss.backward() 时， 会为每个参数计算梯度，并将其存储在 paramter.grad 中， 注意到， paramter.grad 是一个张量， 其会累积求和每次计算得到的梯度。

* 在 Pytorch 中， 只有调用 optimizer.step() 才会进行梯度下降更新网络参数。

我们知道， batch size 与占用显存息息相关，但有时候我们的batch size 又不能设置的太小，这咋办呢？ 答案就是梯度累加。

我们先来看看传统训练：
```python
for i,(feature,target) in enumerate(train_loader):
    outputs = model(feature)  # 前向传播
    loss = criterion(outputs,target)  # 计算损失
    optimizer.zero_grad()   # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 反向传播， 更新网络参数
```
而加入梯度累加之后，代码是这样的：
```python
for i,(features,target) in enumerate(train_loader):
    outputs = model(images)  # 前向传播
    loss = criterion(outputs,target)  # 计算损失
    loss = loss/accumulation_steps   # 可选，如果损失要在训练样本上取平均
    loss.backward()  # 计算梯度
    if((i+1)%accumulation_steps)==0:
        optimizer.step()        # 反向传播，更新网络参数
        optimizer.zero_grad()   # 清空梯度
```
比较来看， 我们发现，梯度累加本质上就是累加 accumulation_steps 个batch 的梯度， 再根据累加的梯度来更新网络参数，以达到类似batch_size 为 accumulation_steps * batch_size 的效果。在使用时，需要注意适当的扩大学习率。

**在Bert的仓库中，就使用了这个Trick，十分实用，简直是我们这种乞丐实验室的良心Trick。 **

### 9、梯度检查点

这个Trick我没用过，毕竟模型还没有那么那么大。

等我用过再更新吧，先把坑挖下。

Reference
## Reference
[1] https://www.zybuluo.com/songying/note/1481006    

[1]科普帖：深度学习中GPU和显存分析

[2]如何在Pytorch中精细化利用显存

[3]GPU捉襟见肘还想训练大批量模型？谁说不可以

[5]PyTorch中在反向传播前为什么要手动将梯度清零？

[6]From zero to research — An introduction to Meta-learning