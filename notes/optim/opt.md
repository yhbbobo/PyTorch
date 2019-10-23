# TORCH.OPTIM 包
[source](https://pytorch.org/docs/stable/optim.html#)   

torch.optim is a package implementing various optimization algorithms. Most commonly used methods are already supported, and the interface is general enough, so that more sophisticated ones can be also easily integrated in the future.  

torch.optim是一个实现各种优化算法的包。已经支持最常用的方法，并且界面足够通用，因此将来可以轻松集成更复杂的方法。
## 一、How to use an optimizer（如何使用一个优化器）
To use torch.optim you have to construct an optimizer object, that will hold the current state and will update the parameters based on the computed gradients.     

要使用，您必须构造一个优化器对象，该对象将保持**当前状态**并将根据**计算的梯度**更新参数。
### 1、Constructing it（构造它）
To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variable s) to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.   

要构造一个优化器，你必须给它一个包含参数的迭代（所有应该是Variables）来优化。然后，您可以指定特定于优化程序的选项，例如学习率，重量衰减等。
* NOTE  
If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it. Parameters of a model after .cuda() will be different objects with those before the call.   
In general, you should make sure that optimized parameters live in consistent locations when optimizers are constructed and used.
* 注意
如果您需要通过.cuda()将模型移动到GPU，请在为其构建优化器之前执行此操作。 .cuda()之后的模型参数与调用之前的参数不同。   
通常，在构造和使用优化程序时，应确保优化参数位于一致的位置。

例：
```python
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)
```
### 2、Per-parameter options（每个参数选项）
Optimizers also support specifying per-parameter options. To do this, instead of passing an iterable of Variables, pass in an iterable of dicts. Each of them will define a separate parameter group, and should contain a params key, containing a list of parameters belonging to it. Other keys should match the keyword arguments accepted by the optimizers, and will be used as optimization options for this group.

Optimizers还支持指定每个参数选项。要做到这一点，不要传递一个可迭代的Variable，而是传递一个可迭代的Variables。它们中的每一个都将定义一个单独的参数组，并且应包含params键，其中包含属于它的参数列表。其他键应与优化程序接受的关键字参数匹配，并将用作此组的优化选项。   
* NOTE
You can still pass options as keyword arguments. They will be used as defaults, in the groups that didn’t override them. This is useful when you only want to vary a single option, while keeping all others consistent between parameter groups.   
您仍然可以将选项作为关键字参数传递。它们将在未覆盖它们的组中用作默认值。当您只想改变单个选项，同时保持参数组之间的所有其他选项保持一致时，这非常有用。

For example, this is very useful when one wants to specify per-layer learning rates:   
例如，当想要指定每层学习速率时，这非常有用：

```python
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```

This means that model.base’s parameters will use the default learning rate of 1e-2, model.classifier’s parameters will use a learning rate of 1e-3, and a momentum of 0.9 will be used for all parameters.   
这意味着model.base的参数将使用1e-2的默认学习速率，model.classifier的参数将使用1e-3的学习速率，0.9的动量将用于所有参数   

### 3、Taking an optimization step
All optimizers implement a step() method, that updates the parameters. It can be used in two ways:   
所有优化器都实现了一个更新参数的方法。它可以以两种方式使用：   
#### optimizer.step()
This is a simplified version supported by most optimizers. The function can be called once the gradients are computed using e.g. backward().   
这是大多数优化器支持的简化版本。一旦使用（例如backward()）计算梯度，就可以调用该函数。 

Example:
```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```
#### optimizer.step(closure)
Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients, compute the loss, and return it.   
一些优化算法，例如Conjugate Gradient和LBFGS需要多次重新评估函数，因此您必须传入一个允许它们重新计算模型的闭包。闭合应清除梯度，计算损失并返回。   

Example:
```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```
## 二、Algorithms
### 1、class torch.optim.Optimizer(params, defaults)
Base class for all optimizers.   
所有优化器的基类。   
#### WARNING
Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs. Examples of objects that don’t satisfy those properties are sets and iterators over values of dictionaries.   
需要将参数指定为具有在运行之间一致的确定性排序的集合。不满足这些属性的对象的示例是字典值的集合和迭代器。

#### Parameters
* params (iterable) – an iterable of torch.Tensor s or dicts. Specifies what Tensors should be optimized.
* defaults – (dict): a dict containing default values of optimization options (used when a parameter group doesn’t specify them).
   
#### 所具有的方法
* add_param_group(param_group)   
Add a param group to the Optimizers param_groups.       
This can be useful when fine tuning a pre-trained network as frozen layers can be made trainable and added to the Optimizer as training progresses.
    * Parameters   
        * param_group (dict) – Specifies what Tensors should be optimized along with group
        * optimization options. (specific) –
* load_state_dict(state_dict)   
Loads the optimizer state.

    * Parameters
        * state_dict (dict) – optimizer state. Should be an object returned from a call to state_dict().

* state_dict()   
Returns the state of the optimizer as a dict.
It contains two entries:   
state - a dict holding current optimization state. Its content
differs between optimizer classes.    
param_groups - a dict containing all parameter groups

* step(closure)
Performs a single optimization step (parameter update).
* Parameters   
closure (callable) – A closure that reevaluates the model and returns the loss. Optional for most optimizers.

* zero_grad()    
Clears the gradients of all optimized torch.Tensor s.   

### 2、torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

### 3、torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)   
### 4、torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
### 5、torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
### 6、torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
### 7、torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)   
### 8、torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
### 9、torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
### 10、torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
### 11、torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
### 12、torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
## 三、How to adjust Learning Rate
torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.    torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based on some validation measurements.
torch.optim.lr_scheduler提供了几种根据时期数调整学习率的方法。允许基于一些验证测量来降低动态学习速率。



### 1、torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

### 2、torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

### 3、torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

### 4、torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

### 5、torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

### 6、torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

### 7、torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

### 8、torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)

### 9、torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1)
