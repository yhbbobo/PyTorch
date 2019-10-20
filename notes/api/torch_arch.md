# Torch V1.3.0 API
## 1、torch
* Tensors（张量）
    * Creation Ops（创建操作）
    * Indexing, Slicing, Joining, Mutating Ops（索引，切片，连接，换位操作）
* Generators（生成器）
* Random sampling（随机采样）
    * In-place random sampling（直接随机采样）
    * Quasi-random sampling（标准随机采样）
* Serialization（序列化）
* Parallelism（并行化）
* Locally disabling gradient computation
* Math operations
    * Pointwise Ops
    * Reduction Ops
    * Comparison Ops
    * Spectral Ops
    * Other Operations
    * BLAS and LAPACK Operations
* Utilities
## 2、torch.nn
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
## 3、torch.nn.functional
* Convolution functions
    * conv1d
    * conv2d
    * conv3d
    * conv_transpose1d
    * conv_transpose2d
    * conv_transpose3d
    * unfold
    * fold
* Pooling functions
    * avg_pool1d
    * avg_pool2d
    * avg_pool3d
    * max_pool1d
    * max_pool2d
    * max_pool3d
    * max_unpool1d
    * max_unpool2d
    * max_unpool3d
    * lp_pool1d
    * lp_pool2d
    * adaptive_max_pool1d
    * adaptive_max_pool2d
    * adaptive_max_pool3d
    * adaptive_avg_pool1d
    * adaptive_avg_pool2d
    * adaptive_avg_pool3d
* Non-linear activation functions
    * threshold
    * relu
    * hardtanh
    * relu6
    * elu
    * selu
    * celu
    * leaky_relu
    * prelu
    * rrelu
    * glu
    * gelu
    * logsigmoid
    * hardshrink
    * tanhshrink
    * softsign
    * softplus
    * softmin
    * softmax
    * softshrink
    * gumbel_softmax
    * log_softmax
    * tanh
    * sigmoid
* Normalization functions
    * batch_norm
    * instance_norm
    * layer_norm
    * local_response_norm
    * normalize
* Linear functions
    * linear
    * bilinear
* Dropout functions
    * dropout
    * alpha_dropout
    * dropout2d
    * dropout3d
* Sparse functions
    * embedding
    * embedding_bag
    * one_hot
* Distance functions
    * pairwise_distance
    * cosine_similarity
    * pdist
* Loss functions
    * binary_cross_entropy
    * binary_cross_entropy_with_logits
    * poisson_nll_loss
    * cosine_embedding_loss
    * cross_entropy
    * ctc_loss
    * hinge_embedding_loss
    * kl_div
    * l1_loss
    * mse_loss
    * margin_ranking_loss
    * multilabel_margin_loss
    * multilabel_soft_margin_loss
    * multi_margin_loss
    * nll_loss
    * smooth_l1_loss
    * soft_margin_loss
    * triplet_margin_loss
* Vision functions
    * pixel_shuffle
    * pad
    * interpolate
    * upsample
    * upsample_nearest
    * upsample_bilinear
    * grid_sample
    * affine_grid
* DataParallel functions (multi-GPU, distributed)
    * data_parallel
## 4、torch.Tensor
* DataType（数据类型）
* torch.Tensor
    * Funtion
    * Property
## 5、torch.autograd
Automatic differentiation package - torch.autograd
* Locally disabling gradient computation
* In-place operations on Tensors
    * In-place correctness checks
* Variable (deprecated)
* Tensor autograd functions
* Function
* Numerical gradient checking
* Profiler
* Anomaly detection
## 6、torch.cuda
* Random Number Generator
* Communication collectives
* Streams and events
* Memory management
* NVIDIA Tools Extension (NVTX)
## 7、torch.distributed
Distributed communication package - torch.distributed
* Backends
    * Backends that come with PyTorch
    * Which backend to use?
* Common environment variables
    * Choosing the network interface to use
    * Other NCCL environment variables
* Basics
* Initialization
    * TCP initialization
    * Shared file-system initialization
    * Environment variable initialization
* Groups
* Point-to-point communication
* Synchronous and asynchronous collective operations
* Collective functions
* Multi-GPU collective functions
* Launch utility
* Spawn utility
## 8、torch.distributions
Probability distributions - torch.distributions
* Score function
* Pathwise derivative
* Distribution
* ExponentialFamily
* Bernoulli
* Beta
* Binomial
* Categorical
* Cauchy
* Chi2
* Dirichlet
* Exponential
* FisherSnedecor
* Gamma
* Geometric
* Gumbel
* HalfCauchy
* HalfNormal
* Independent
* Laplace
* LogNormal
* LowRankMultivariateNormal
* Multinomial
* MultivariateNormal
* NegativeBinomial
* Normal
* OneHotCategorical
* Pareto
* Poisson
* RelaxedBernoulli
* LogitRelaxedBernoulli
* RelaxedOneHotCategorical
* StudentT
* TransformedDistribution
* Uniform
* Weibull
* KL Divergence
* Transforms
* Constraints
* Constraint Registry
## 9、torch.hub
* Publishing models
    * How to implement an entrypoint?
    * Important Notice
* Loading models from Hub
    * Running a loaded model:
    * Where are my downloaded models saved?
    * Caching logic
    * Known limitations:
## 10、torch.jit
TorchScript
* Creating TorchScript Code
* Mixing Tracing and Scripting
* Migrating to PyTorch 1.2 Recursive Scripting API
    * Modules
    * Functions
    * TorchScript Classes
    * Attributes
        * Python 2
    * Constants
    * Variables
* TorchScript Language Reference
    * Types
        * Default Types
        * Optional Type Refinement
        * TorchScript Classes
        * Named Tuples
    * Expressions
        * Literals
            * List Construction
            * Tuple Construction
            * Dict Construction
        * Variables
        * Arithmetic Operators
        * Comparison Operators
        * Logical Operators
        * Subscripts and Slicing
        * Function Calls
        * Method Calls
        * Ternary Expressions
        * Casts
        * Accessing Module Parameters
    * Statements
        * Simple Assignments
        * Pattern Matching Assignments
        * Print Statements
        * If Statements
        * While Loops
        * For loops with range
        * For loops over tuples
        * For loops over constant nn.ModuleList
        * Break and Continue
        * Return
    * Variable Resolution
    * Use of Python Values
        * Functions
        * Attribute Lookup On Python Modules
        * Python-defined Constants
        * Module Attributes
    * Debugging
        * Disable JIT for Debugging
        * Inspecting Code
        * Interpreting Graphs
        * Tracing Edge Cases
        * Automatic Trace Checking
        * Tracer Warnings
    * Builtin Functions
* Frequently Asked Questions
## 11、torch.nn.init
* calculate_gain
* uniform_
* normal_
* constant_
* ones_
* zeros_
* eye_
* dirac_
* xavier_uniform_
* xavier_normal_
* kaiming_uniform_
* kaiming_normal_
* orthogonal_
* sparse_
## 12、torch.onnx
* Example: End-to-end AlexNet from PyTorch to ONNX
* Tracing vs Scripting
* Limitations
* Supported operators
* Adding support for operators
    * ATen operators
    * Non-ATen operators
    * Custom operators
* Frequently Asked Questions
* Functions
## 13、torch.optim
* How to use an optimizer
    * Constructing it
    * Per-parameter options
    * Taking an optimization step
        * optimizer.step()
        * optimizer.step(closure)
* Algorithms
* How to adjust Learning Rate
* Quantization
## 14、Quantization
* Introduction to Quantization
* Quantized Tensors
* Operation coverage
* Quantized torch.Tensor operations
    * torch.nn.intrinsic
    * torch.nn.qat
    * torch.quantization
    * torch.nn.quantized
    * torch.nn.quantized.dynamic
    * torch.nn.quantized.functional
    * Quantized dtypes and quantization schemes
* Quantization Workflows
* Model Preparation for Quantization
* torch.quantization
    * Top-level quantization APIs
    * Preparing model for quantization
    * Utility functions
    * Observers
    * Debugging utilities
* torch.nn.instrinsic
    * ConvBn2d
    * ConvBnReLU2d
    * ConvReLU2d
    * LinearReLU
* torch.nn.instrinsic.qat
    * ConvBn2d
    * ConvBnReLU2d
    * ConvReLU2d
    * LinearReLU
*  torch.nn.intrinsic.quantized
    * ConvReLU2d
    * LinearReLU
    * torch.nn.qat
    * Conv2d
    * Linear
* torch.nn.quantized
    * Functional interface
    * ReLU
    * ReLU6
    * Conv2d
    * FloatFunctional
    * QFunctional
    * Quantize
    * DeQuantize
    * Linear
* torch.nn.quantized.dynamic
    * Linear
    * LSTM
## 15、torch.random
* Random Number Generator
## 16、torch.sparse
* Functions
## 17、torch.Storage
## 18、torch.utils.bottleneck
## 19、torch.utils.checkpoint
## 20、torch.utils.cpp_extension
## 21、torch.utils.data
* Dataset Types
    * Map-style datasets
    * Iterable-style datasets
* Data Loading Order and Sampler
* Loading Batched and Non-Batched Data
    * Automatic batching (default)
    * Disable automatic batching
    * Working with collate_fn
* Single- and Multi-process Data Loading
    * Single-process data loading (default)
    * Multi-process data loading
        * Platform-specific behaviors
        * Randomness in multi-process data loading
* Memory Pinning
## 22、torch.utils.dlpack
## 23、torch.utils.model_zoo
## 24、torch.utils.tensorboard
## 25、Type Info
* torch.finfo
* torch.iinfo
## 26、Named Tensors
* Creating named tensors
* Named dimensions
* Name propagation semantics
    * match semantics
    * Basic name inference rules
* Explicit alignment by names
* Manipulating dimensions
* Autograd support
* Currently supported operations and subsystems
    * Operators
    * Subsystems
* Named tensor API reference
* Named Tensors operator coverage
## 27、Named Tensors operator coverage
* Keeps input names
* Removes dimensions
* Unifies names from inputs
* Permutes dimensions
* Contracts away dims
* Factory functions
* out function and in-place variants
## 28、torch.__config__

# torchvision Reference
* torchvision.datasets
    * MNIST
    * Fashion-MNIST
    * KMNIST
    * EMNIST
    * QMNIST
    * FakeData
    * COCO
    * LSUN
    * ImageFolder
    * DatasetFolder
    * ImageNet
    * CIFAR
    * STL10
    * SVHN
    * PhotoTour
    * SBU
    * Flickr
    * VOC
    * Cityscapes
    * SBD
    * USPS
    * Kinetics-400
    * HMDB51
    * UCF101
* torchvision.io
    * Video
* torchvision.models
    * Classification
    * Semantic Segmentation
    * Object Detection, Instance Segmentation * and Person Keypoint Detection
    * Video classification
* torchvision.ops
    * torchvision.transforms
    * Transforms on PIL Image
    * Transforms on torch.*Tensor
    * Conversion Transforms
    * Generic Transforms
    * Functional Transforms
* torchvision.utils