# torch.nn package
## 一、torch.nn包目录结构
* [Parameters](notes/api/torch_nn_parameter.md)
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
* Loss functions
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
