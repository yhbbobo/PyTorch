# Loss functions
* loss functions
    * L1Loss
    * MSELoss
    * CrossEntropyLoss
    * CTCLoss
    * NLLLoss
    * PoissonNLLLoss
    * KLDivLoss
    * BCELoss
    * BCEWithLogitsLoss
    * MarginRankingLoss
    * HingeEmbeddingLoss
    * MultiLabelMarginLoss
    * SmoothL1Loss
    * SoftMarginLoss
    * MultiLabelSoftMarginLoss
    * CosineEmbeddingLoss
    * MultiMarginLoss
    * TripletMarginLoss

损失函数通过torch.nn包实现  
### 基本用法
* criterion = LossCriterion() #构造函数有自己的参数
* loss = criterion(x, y) #调用标准时也有参数
## 一、L1Loss
`class torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')`     
* 功能：  
计算output和target之差的绝对值，可选返回同维度的tensor或者是一个标量。
* 计算公式：   
![](../../imgs/api/02.png)   
* 参数：
    * reduce(bool)- 返回值是否为标量，默认为True
    * size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
* 实例：
[🔗](https://github.com/fusimeng/PyTorch_Tutorial/blob/master/Code/3_optimizer/3_1_lossFunction/1_L1Loss.py)

## 二、MSELoss
`class torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`     
* 功能：   
计算output和target之差的平方，可选返回同维度的tensor或者是一个标量。
* 计算公式：  
![](../../imgs/api/03.png)   
* 参数：   
    * reduce(bool)- 返回值是否为标量，默认为True
    * size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
* 实例：   
[link](https://github.com/TingsongYu/PyTorch_Tutorial/blob/master/Code/3_optimizer/3_1_lossFunction/2_MSELoss.py)   

## 三、CrossEntropyLoss
[极大似然估计](mle.md)   
[交叉熵损失函数](crossentropy.md)   

`class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')`    
* 功能：  
将输入经过softmax激活函数之后，再计算其与target的交叉熵损失。即该方法将nn.LogSoftmax()和 nn.NLLLoss()进行了结合。严格意义上的交叉熵损失函数应该是nn.NLLLoss()。

* 补充：小谈交叉熵损失函数   
交叉熵损失(cross-entropy Loss) 又称为对数似然损失(Log-likelihood Loss)、对数损失；二分类时还可称之为逻辑斯谛回归损失(Logistic Loss)。交叉熵损失函数表达式为 L = - sigama(y_i * log(x_i))。pytroch这里不是严格意义上的交叉熵损失函数，而是先将input经过softmax激活函数，将向量“归一化”成概率形式，然后再与target计算严格意义上交叉熵损失。   
在多分类任务中，经常采用softmax激活函数+交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。所以需要softmax激活函数将一个向量进行“归一化”成概率分布的形式，再采用交叉熵损失函数计算loss。
再回顾PyTorch的CrossEntropyLoss()，官方文档中提到时将nn.LogSoftmax()和 nn.NLLLoss()进行了结合，nn.LogSoftmax() 相当于激活函数 ， nn.NLLLoss()是损失函数，将其结合，完整的是否可以叫做softmax+交叉熵损失函数呢？   
* 计算公式：   
![](../../imgs/api/04.png)   

* 参数：  
    * weight(Tensor)- 为每个类别的loss设置权值，常用于类别不均衡问题。weight必须是float类型的tensor，其长度要于类别C一致，即每一个类别都要设置有weight。带weight的计算公式：
    * size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
    * reduce(bool)- 返回值是否为标量，默认为True
    * ignore_index(int)- 忽略某一类别，不计算其loss，其loss会为0，并且，在采用size_average时，不会计算那一类的loss，除的时候的分母也不会统计那一类的样本。
* 实例：   
[link](https://github.com/TingsongYu/PyTorch_Tutorial/blob/master/Code/3_optimizer/3_1_lossFunction/3_CrossEntropyLoss.py)   
* 补充：   
output不仅可以是向量，还可以是图片，即对图像进行像素点的分类，这个例子可以从NLLLoss()中看到，这在图像分割当中很有用。

## 四、CTCLoss
CLASStorch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
## 五、NLLLoss
CLASStorch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
## 六、PoissonNLLLoss
CLASStorch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
## 七、KLDivLoss
CLASStorch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean')
## 八、BCELoss
CLASStorch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
## 九、BCEWithLogitsLoss
CLASStorch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
## 十、MarginRankingLoss
CLASStorch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
## 十一、HingeEmbeddingLoss
CLASStorch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')
## 十二、MultiLabelMarginLoss
CLASStorch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
## 十三、SmoothL1Loss
CLASStorch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')
## 十四、SoftMarginLoss
CLASStorch.nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')
## 十五、MultiLabelSoftMarginLoss
CLASStorch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='mean')
## 十六、CosineEmbeddingLoss
CLASStorch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
## 十七、MultiMarginLoss
CLASStorch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
## 十八、TripletMarginLoss
CLASStorch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')