# torch.nn 包
包：`torch.nn`   

## 目录结构
torch.nn
* Parameters
* Containers
    * Module
    * Sequential
    * ModuleList
    * ModuleDict
    * ParameterList
    * ParameterDict
* Convolution layers
    * Conv1d
    * Conv2d
    * Conv3d
    * ConvTranspose1d
    * ConvTranspose2d
    * ConvTranspose3d
    * Unfold
    * Fold
* Pooling layers
    * MaxPool1d
    * MaxPool2d
    * MaxPool3d
    * MaxUnpool1d
    * MaxUnpool2d
    * MaxUnpool3d
    * AvgPool1d
    * AvgPool2d
    * AvgPool3d
    * FractionalMaxPool2d
    * LPPool1d
    * LPPool2d
    * AdaptiveMaxPool1d
    * AdaptiveMaxPool2d
    * AdaptiveMaxPool3d
    * AdaptiveAvgPool1d
    * AdaptiveAvgPool2d
    * AdaptiveAvgPool3d
* Padding layers
    * ReflectionPad1d
    * ReflectionPad2d
    * ReplicationPad1d
    * ReplicationPad2d
    * ReplicationPad3d
    * ZeroPad2d
    * ConstantPad1d
    * ConstantPad2d
    * ConstantPad3d
* Non-linear activations (weighted sum, nonlinearity)
    * ELU
    * Hardshrink
    * Hardtanh
    * LeakyReLU
    * LogSigmoid
    * MultiheadAttention
    * PReLU
    * ReLU
    * ReLU6
    * RReLU
    * SELU
    * CELU
    * Sigmoid
    * Softplus
    * Softshrink
    * Softsign
    * Tanh
    * Tanhshrink
    * Threshold
* Non-linear activations (other)
    * Softmin
    * Softmax
    * Softmax2d
    * LogSoftmax
    * AdaptiveLogSoftmaxWithLoss
* Normalization layers
    * BatchNorm1d
    * BatchNorm2d
    * BatchNorm3d
    * GroupNorm
    * SyncBatchNorm
    * InstanceNorm1d
    * InstanceNorm2d
    * InstanceNorm3d
    * LayerNorm
    * LocalResponseNorm
* Recurrent layers
    * RNN
    * LSTM
    * GRU
    * RNNCell
    * LSTMCell
    * GRUCell
* Transformer layers
    * Transformer
    * TransformerEncoder
    * TransformerDecoder
    * TransformerEncoderLayer
    * TransformerDecoderLayer
* Linear layers
    * Identity
    * Linear
    * Bilinear
* Dropout layers
    * Dropout
    * Dropout2d
    * Dropout3d
    * AlphaDropout
* Sparse layers
    * Embedding
    * EmbeddingBag
* Distance functions
    * CosineSimilarity
    * PairwiseDistance
* [Loss functions](loss.md)
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
* Vision layers
    * PixelShuffle
    * Upsample
    * UpsamplingNearest2d
    * UpsamplingBilinear2d
* DataParallel layers (multi-GPU, distributed)
    * DataParallel
    * DistributedDataParallel
* Utilities
    * clip_grad_norm_
    * clip_grad_value_
    * parameters_to_vector
    * vector_to_parameters
    * weight_norm
    * remove_weight_norm
    * spectral_norm
    * remove_spectral_norm
    * PackedSequence
    * pack_padded_sequence
    * pad_packed_sequence
    * pad_sequence
    * pack_sequence
    * Flatten
* Quantized Functions
## 一、Parameters   
[en](https://pytorch.org/docs/stable/nn.html#parameters)/[cn](https://pytorch.apachecn.org/docs/1.2/nn.html#parameters%EF%BC%88%E5%8F%82%E6%95%B0%EF%BC%89)/[source](https://pytorch.org/docs/stable/_modules/torch/nn/parameter.html#Parameter)
### 1、CLASS torch.nn.Parameter
## 二、Containers
[en](https://pytorch.org/docs/stable/nn.html#containers)/[cn](https://pytorch.apachecn.org/docs/1.2/nn.html#containers%EF%BC%88%E5%AE%B9%E5%99%A8%EF%BC%89)/[source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module)
### 1、Module
* CLASS torch.nn.Module(object)
    * add_module(name, module)
    * apply(fn)
    * buffers(recurse=True)
    * children()
    * cpu()
    * cuda(device=None)
    * double()
    * dump_patches = FALSE
    * eval()
    * extra_repr()
    * float()
    * forward(*input)
    * half()
    * load_state_dict(state_dict, strict=True)
    * modules()
    * named_buffers(prefix='', recurse=True)
    * named_children()
    * named_modules(memo=None, prefix='')
    * named_parameters(prefix='', recurse=True)
    * parameters(recurse=True)
    * register_backward_hook(hook)
    * register_buffer(name, tensor)
    * register_forward_hook(hook)
    * register_forward_pre_hook(hook)
    * register_parameter(name, param)
    * requires_grad_(requires_grad=True)
    * state_dict(destination=None, prefix='', keep_vars=False)
    * to(*args, **kwargs)
    * train(mode=True)
    * type(dst_type)
### 2、CLASS torch.nn.Sequential(*args）
### 3、 CLASS torch.nn.ModuleList(modules=None)
* append(module)
* extend(modules)
* insert(index, module)
### 4、CLASS torch.nn.ModuleDict(modules=None)
* clear()
* items()
* keys()
* pop(key)
* update(modules)
* values()
### 5、CLASS torch.nn.ParameterList(parameters=None)
* append(parameter)
* extend(parameters)
### 6、CLASS torch.nn.ParameterDict(parameters=None)
* clear()
* items()
* keys()
* pop(key)
* update(modules)
* values()
## 三、Convolution layers
[en](https://pytorch.org/docs/stable/nn.html#convolution-layers)/[cn](https://pytorch.apachecn.org/docs/1.2/nn.html#convolution-layers-%E5%8D%B7%E7%A7%AF%E5%B1%82)/[source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d)
### 1、Conv1d
* CLASStorch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
### 2、Conv2d
* torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
### 3、Conv3d
CLASStorch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
### 4、ConvTranspose1d
* CLASStorch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
### 5、ConvTranspose2d
* CLASStorch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
### 6、ConvTranspose3d
* CLASStorch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
### 7、Unfold
* CLASStorch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
### 8、Fold
* CLASStorch.nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)
## 四、Pooling layers
[en](https://pytorch.org/docs/stable/nn.html#pooling-layers)/[cn](https://pytorch.apachecn.org/docs/1.2/nn.html#%E6%B1%A0%E5%8C%96%E5%B1%82%EF%BC%88pooling-layers%EF%BC%89)/[source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html#MaxPool1d)
### 1、MaxPool1d
* CLASStorch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
### 2、MaxPool2d
* CLASStorch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
### 3、MaxPool3d
* CLASStorch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
### 4、MaxUnpool1d
* CLASStorch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
### 5、MaxUnpool2d
* CLASStorch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
### 6、MaxUnpool3d
* CLASStorch.nn.MaxUnpool3d(kernel_size, stride=None, padding=0)
### 7、AvgPool1d
* CLASStorch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
### 8、AvgPool2d
* CLASStorch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
### 9、AvgPool3d
* CLASStorch.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
### 10、FractionalMaxPool2d
* CLASStorch.nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
### 11、LPPool1d
* CLASStorch.nn.LPPool1d(norm_type, kernel_size, stride=None, ceil_mode=False)
### 12、 LPPool2d
* CLASStorch.nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)
### 13、AdaptiveMaxPool1d
* CLASStorch.nn.AdaptiveMaxPool1d(output_size, return_indices=False)
### 14、AdaptiveMaxPool2d
* CLASStorch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
### 15、AdaptiveMaxPool3d
* CLASStorch.nn.AdaptiveMaxPool3d(output_size, return_indices=False)
### 16、AdaptiveAvgPool1d
* CLASStorch.nn.AdaptiveAvgPool1d(output_size)
### 17、AdaptiveAvgPool2d
* CLASStorch.nn.AdaptiveAvgPool2d(output_size)
### 18、AdaptiveAvgPool3d
* CLASStorch.nn.AdaptiveAvgPool3d(output_size)
## 五、Padding layers
[en](https://pytorch.org/docs/stable/nn.html#padding-layers)/[cn](https://pytorch.apachecn.org/docs/1.2/nn.html#%E5%A1%AB%E5%85%85%E5%B1%82%EF%BC%88padding-layers%EF%BC%89)/[source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/padding.html#ReflectionPad1d)
### 1、ReflectionPad1d
* CLASStorch.nn.ReflectionPad1d(padding)
### 2、ReflectionPad2d
* CLASStorch.nn.ReflectionPad2d(padding)
### 3、ReplicationPad1d
* CLASStorch.nn.ReplicationPad1d(padding)
### 4、ReplicationPad2d
* CLASStorch.nn.ReplicationPad2d(padding)
### 5、ReplicationPad3d
* CLASStorch.nn.ReplicationPad3d(padding)
### 6、ZeroPad2d
* CLASStorch.nn.ZeroPad2d(padding)
### 7、ConstantPad1d
* CLASStorch.nn.ConstantPad1d(padding, value)
### 8、ConstantPad2d
* CLASStorch.nn.ConstantPad2d(padding, value)
### 9、ConstantPad3d
* CLASStorch.nn.ConstantPad3d(padding, value)
## 六、Non-linear activations (weighted sum, nonlinearity)
[en](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)/[cn](https://pytorch.apachecn.org/docs/1.2/nn.html#%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%BF%80%E6%B4%BB%E5%8A%A0%E6%9D%83%E6%B1%82%E5%92%8C%EF%BC%8C%E9%9D%9E%E7%BA%BF%E6%80%A7--non-linear-activations-weighted-sum-nonlinearity-)/[source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ELU)
### 1、ELU
* CLASStorch.nn.ELU(alpha=1.0, inplace=False)
### 2、Hardshrink
* CLASStorch.nn.Hardshrink(lambd=0.5)
### 3、Hardtanh
* CLASStorch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None)
### 4、LeakyReLU
* CLASStorch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
### 5、LogSigmoid
* CLASStorch.nn.LogSigmoid
### 6、MultiheadAttention
* CLASStorch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
### 7、PReLU
* CLASStorch.nn.PReLU(num_parameters=1, init=0.25)
### 8、ReLU
* CLASStorch.nn.ReLU(inplace=False)
### 9、ReLU6
* CLASStorch.nn.ReLU6(inplace=False)
### 10、RReLU
* CLASStorch.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)
### 11、SELU
* CLASStorch.nn.SELU(inplace=False)
### 12、CELU
* CLASStorch.nn.CELU(alpha=1.0, inplace=False)
### 13、Sigmoid
* CLASStorch.nn.Sigmoid
### 14、Softplus
* CLASStorch.nn.Softplus(beta=1, threshold=20)
### 15、Softshrink
* CLASStorch.nn.Softshrink(lambd=0.5)
### 16、Softsign
* CLASStorch.nn.Softsign
### 17、Tanh
* CLASStorch.nn.Tanh
### 18、Tanhshrink
* CLASStorch.nn.Tanhshrink
### 19、Threshold
* CLASStorch.nn.Threshold(threshold, value, inplace=False)
## 七、Non-linear activations (other)
[en](https://pytorch.org/docs/stable/nn.html#non-linear-activations-other)/[cn](https://pytorch.apachecn.org/docs/1.2/nn.html#non-linear-activations-other)/[source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Softmin)
### 1、Softmin
* CLASStorch.nn.Softmin(dim=None)
### 2、Softmax
* CLASStorch.nn.Softmax(dim=None)
### 3、Softmax2d
* CLASStorch.nn.Softmax2d
### 4、LogSoftmax
* CLASStorch.nn.LogSoftmax(dim=None)
### 5、AdaptiveLogSoftmaxWithLoss
* CLASStorch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False)
    * log_prob(input)
    * predict(input)
## 八、Normalization layers
### 1、BatchNorm1d
CLASStorch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
### 2、BatchNorm2d
CLASStorch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
### 3、BatchNorm3d
CLASStorch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
### 4、GroupNorm
CLASStorch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)
### 5、SyncBatchNorm
CLASStorch.nn.SyncBatchNorm(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None)
### 6、InstanceNorm1d
CLASStorch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
### 7、InstanceNorm2d
CLASStorch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
### 8、InstanceNorm3d
CLASStorch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
### 9、LayerNorm
CLASStorch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
### 10、LocalResponseNorm
CLASStorch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)
## 九、Recurrent layers
### 1、RNN
CLASStorch.nn.RNN(*args, **kwargs)
### 2、LSTM
CLASStorch.nn.LSTM(*args, **kwargs)
### 3、GRU
CLASStorch.nn.GRU(*args, **kwargs)
### 4、RNNCell
CLASStorch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
### 5、LSTMCell
CLASStorch.nn.LSTMCell(input_size, hidden_size, bias=True)
### 6、GRUCell
CLASStorch.nn.GRUCell(input_size, hidden_size, bias=True)
## 十、Transformer layers
### 1、Transformer
CLASStorch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)
### 2、TransformerEncoder
CLASStorch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
### 3、TransformerDecoder
CLASStorch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
### 4、TransformerEncoderLayer
CLASStorch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
### 5、TransformerDecoderLayer
CLASStorch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
## 十一、Linear layers
### 1、Identity
CLASStorch.nn.Identity(*args, **kwargs)
### 2、Linear
CLASStorch.nn.Linear(in_features, out_features, bias=True)
### 3、Bilinear
CLASStorch.nn.Bilinear(in1_features, in2_features, out_features, bias=True)
## 十二、Dropout layers
### 1、Dropout
CLASStorch.nn.Dropout(p=0.5, inplace=False)
### 2、Dropout2d
CLASStorch.nn.Dropout2d(p=0.5, inplace=False)
### 3、Dropout3d
CLASStorch.nn.Dropout3d(p=0.5, inplace=False)
### 4、AlphaDropout
CLASStorch.nn.AlphaDropout(p=0.5, inplace=False)
## 十三、Sparse layers
### 1、Embedding
CLASStorch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
### 2、EmbeddingBag
CLASStorch.nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None)
## 十四、Distance functions
### 1、CosineSimilarity
CLASStorch.nn.CosineSimilarity(dim=1, eps=1e-08)
### 2、PairwiseDistance
CLASStorch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
## 十五、Loss functions

## 十六、Vision layers
### 1、PixelShuffle
CLASStorch.nn.PixelShuffle(upscale_factor)
### 2、Upsample
CLASStorch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
### 3、UpsamplingNearest2d
CLASStorch.nn.UpsamplingNearest2d(size=None, scale_factor=None)
### 4、UpsamplingBilinear2d
CLASStorch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)
## 十七、DataParallel layers (multi-GPU, distributed)
### 1、DataParallel
CLASStorch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
### 2、DistributedDataParallel
CLASStorch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, process_group=None, bucket_cap_mb=25, find_unused_parameters=False, check_reduction=False)
## 十八、Utilities
### 1、clip_grad_norm_
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2)
### 2、clip_grad_value_
torch.nn.utils.clip_grad_value_(parameters, clip_value)
### 3、parameters_to_vector
torch.nn.utils.parameters_to_vector(parameters)
vector_to_parameters
torch.nn.utils.vector_to_parameters(vec, parameters)
### 4、weight_norm
torch.nn.utils.weight_norm(module, name='weight', dim=0)
remove_weight_norm
torch.nn.utils.remove_weight_norm(module, name='weight')
### 5、spectral_norm
torch.nn.utils.spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None)
### 6、remove_spectral_norm
torch.nn.utils.remove_spectral_norm(module, name='weight')
### 7、PackedSequence
torch.nn.utils.rnn.PackedSequence(data, batch_sizes=None, sorted_indices=None, unsorted_indices=None)
### 8、pack_padded_sequence
torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
### 9、pad_packed_sequence
torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)
### 10、pad_sequence
torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0)
### 11、pack_sequence
torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=True)
### 12、Flatten
CLASStorch.nn.Flatten(start_dim=1, end_dim=-1)
## 十九、Quantized Functions













