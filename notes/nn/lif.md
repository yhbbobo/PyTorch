# PyTorch模型加载/参数初始化/Finetune
在应用过程中不可避免需要使用Finetune/参数初始化/模型加载等。   
## 一、[模型保存/加载](load_pre.md)
### 1.所有模型参数
训练过程中，有时候会由于各种原因停止训练，这时候我们训练过程中就需要
注意将每一轮epoch的模型保存(一般保存最好模型与当前轮模型)。
一般使用pytorch里面推荐的保存方法。该方法保存的是模型的参数。
```
#保存模型到checkpoint.pth.tar
torch.save(model.module.state_dict(), ‘checkpoint.pth.tar’)
```
对应的加载模型方法为(这种方法需要先反序列化模型获取参数字典，
因此必须先load模型，再load_state_dict)：
```
mymodel.load_state_dict(torch.load(‘checkpoint.pth.tar’))
```
有了上面的保存后，现以一个例子说明如何在inference AND/OR resume train使用。
```
#保存模型的状态，可以设置一些参数，后续可以使用
state = {'epoch': epoch + 1,#保存的当前轮数
         'state_dict': mymodel.state_dict(),#训练好的参数
         'optimizer': optimizer.state_dict(),#优化器参数,为了后续的resume
         'best_pred': best_pred#当前最好的精度
          ,....,...}

#保存模型到checkpoint.pth.tar
torch.save(state, ‘checkpoint.pth.tar’)
#如果是best,则复制过去
if is_best:
    shutil.copyfile(filename, directory + 'model_best.pth.tar')

checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])#模型参数
optimizer.load_state_dict(checkpoint['optimizer'])#优化参数
epoch = checkpoint['epoch']#epoch，可以用于更新学习率等

#有了以上的东西，就可以继续重新训练了，也就不需要担心停止程序重新训练。
train/eval
....
....
```
上面是pytorch建议使用的方法，当然还有第二种方法。这种方法灵活性不高，不推荐。
```
#保存
torch.save(mymodel,‘checkpoint.pth.tar’)

#加载
mymodel = torch.load(‘checkpoint.pth.tar’)
```
### 2.部分模型参数
在很多时候，我们加载的是已经训练好的模型，而训练好的模型可能与
我们定义的模型不完全一样，而我们只想使用一样的那些层的参数。

有几种解决方法：

##### （1）直接在训练好的模型开始搭建自己的模型，就是先加载训练好的模型，然后再它基础上定义自己的模型；
```
model_ft = models.resnet18(pretrained=use_pretrained)
self.conv1 = model_ft.conv1
self.bn = model_ft.bn
... ...
```
##### (2) 自己定义好模型，直接加载模型
```
#第一种方法：
mymodelB = TheModelBClass(*args, **kwargs)
# strict=False，设置为false,只保留键值相同的参数
mymodelB.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

#第二种方法：
# 加载模型
model_pretrained = models.resnet18(pretrained=use_pretrained)

# mymodel's state_dict，
# 如：  conv1.weight 
#     conv1.bias  
mymodelB_dict = mymodelB.state_dict()

# 将model_pretrained的建与自定义模型的建进行比较，剔除不同的
pretrained_dict = {k: v for k, v in model_pretrained.items() if k in mymodelB_dict}
# 更新现有的model_dict
mymodelB_dict.update(pretrained_dict)

# 加载我们真正需要的state_dict
mymodelB.load_state_dict(mymodelB_dict)

# 方法2可能更直观一些
```
## 二、参数初始化
第二个问题是参数初始化问题，在很多代码里面都会使用到，
毕竟不是所有的都是有预训练参数。这时就需要对不是与预训练参数进行初始化。
pytorch里面的每个Tensor其实是对Variabl的封装，其包含data、grad等接口，因此可以用这些接口直接赋值。这里也提供了怎样把其他框架(caffe/tensorflow/mxnet/gluonCV等)训练好的模型参数直接赋值给pytorch.其实就是对data直接赋值。

pytorch提供了初始化参数的方法：
```
 def weight_init(m):
    if isinstance(m,nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0,math.sqrt(2./n))
    elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
```
但一般如果没有很大需求初始化参数，也没有问题(不确定性能是否有影响的情况下)，pytorch内部是有默认初始化参数的。

## 三、Fintune
最后就是精调了，我们平时做实验，至少backbone是用预训练的模型，将其用作特征提取器，或者在它上面做精调。

用于**特征提取**的时候，要求特征提取部分参数不进行学习，而pytorch提供了requires_grad参数用于确定是否进去梯度计算，也即是否更新参数。以下以minist为例，用resnet18作特征提取：
```
#加载预训练模型
model = torchvision.models.resnet18(pretrained=True)

#遍历每一个参数，将其设置为不更新参数，即不学习
for param in model.parameters():
    param.requires_grad = False

# 将全连接层改为mnist所需的10类，注意：这样更改后requires_grad默认为True
model.fc = nn.Linear(512, 10)

# 优化
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)         
```
用于**全局精调**时，我们一般对不同的层需要设置不同的学习率，预训练的层学习率小一点，其他层大一点。这要怎么做呢？
```
# 加载预训练模型
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)

# 参考：https://blog.csdn.net/u012759136/article/details/65634477
ignored_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

# 对不同参数设置不同的学习率
params_list = [{'params': base_params, 'lr': 0.001},]
params_list.append({'params': model.fc.parameters(), 'lr': 0.01})

optimizer = torch.optim.SGD(params_list,
                    0.001，
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
```

最后整理一下目前，pytorch预训练的基础模型：   
（1）torchvision   
torchvision里面已经提供了不同的预训练模型，一般也够用了。   
https://github.com/pytorch/vision/tree/master/torchvision/models   
(2)其他预训练好的模型，如，SENet/NASNet等。   
https://github.com/Cadene/pretrained-models.pytorch    
(3)gluonCV转pytorch的模型，包括，分类网络，分割网络等，这里的精度均比其他框架高几个百分点。   
https://github.com/zhanghang1989/gluoncv-torch   



**-----------------------------------------------------------------------------**   
# Reference   
[1] https://zhuanlan.zhihu.com/p/48524007
