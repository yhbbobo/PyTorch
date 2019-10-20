# torch.nn.funtional 包

## 目录结构
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