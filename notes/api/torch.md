# torch
* [官方API](https://pytorch.org/docs/stable/torch.html)   
* [中文API](https://pytorch.apachecn.org/docs/1.2/torch.html)    


类：`torch`   
源码：[🔗](https://pytorch.org/docs/stable/_modules/torch.html)    
## 目录结构
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

## 一、Tensor [en](https://pytorch.org/docs/stable/torch.html#module-torch)/[cn](https://pytorch.apachecn.org/docs/1.2/torch.html)   
功能|参数含义|返回类型 ，参考en/cn. 
* torch.is_tensor(obj)
* torch.is_storage(obj)
* torch.is_floating_point(input) -> (bool)
* torch.set_default_dtype(d)
* torch.get_default_dtype() → torch.dtype
* torch.set_default_tensor_type(t)
* torch.numel(input) → int
* torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
* torch.set_flush_denormal(mode) → bool
### 1、Creation Ops
* torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor
* torch.sparse_coo_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False) → Tensor
* torch.as_tensor(data, dtype=None, device=None) → Tensor
* torch.as_strided(input, size, stride, storage_offset=0) → Tensor
* torch.from_numpy(ndarray) → Tensor
* torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
* torch.ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.ones_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
* torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.range(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor
* torch.empty_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
* torch.empty_strided(size, stride, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) → Tensor
* torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.full_like(input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) → Tensor
### 2、Indexing, Slicing, Joining, Mutating Ops
* torch.cat(tensors, dim=0, out=None) → Tensor
* torch.chunk(input, chunks, dim=0) → List of Tensors
* torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor
* torch.index_select(input, dim, index, out=None) → Tensor
* torch.masked_select(input, mask, out=None) → Tensor
* torch.narrow(input, dim, start, length) → Tensor
* torch.nonzero(input, *, out=None, as_tuple=False) → LongTensor or tuple of LongTensors
* torch.reshape(input, shape) → Tensor
* torch.split(tensor, split_size_or_sections, dim=0)
* torch.squeeze(input, dim=None, out=None) → Tensor
* torch.stack(tensors, dim=0, out=None) → Tensor
* torch.t(input) → Tensor
* torch.take(input, index) → Tensor
* torch.transpose(input, dim0, dim1) → Tensor
* torch.unbind(input, dim=0) → seq
* torch.unsqueeze(input, dim, out=None) → Tensor
* torch.where()
## 二、Generators [en](https://pytorch.org/docs/stable/torch.html#generators)/cn
### 1、torch._C.Generator(device='cpu') → Generator
* device
* get_state() → Tensor
* initial_seed() → int
* manual_seed(seed) → Generator
* seed() → int
* set_state(new_state) → void
## 三、Random sampling [en](https://pytorch.org/docs/stable/torch.html#random-sampling)/[cn](https://pytorch.apachecn.org/docs/1.2/torch.html#random-sampling-%E9%9A%8F%E6%9C%BA%E9%87%87%E6%A0%B7)
* torch.seed()
* torch.manual_seed(seed)
* torch.initial_seed()
* torch.get_rng_state()
* torch.set_rng_state(new_state)
* torch.default_generator Returns the default CPU torch.Generator
* torch.bernoulli(input, *, generator=None, out=None) → Tensor
* torch.multinomial(input, num_samples, replacement=False, out=None) → LongTensor
* torch.normal()
* torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.rand_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
* torch.randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.randint_like(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.randn_like(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor
* torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) → LongTensor
### 1、In-place random sampling
### 2、Quasi-random sampling
* class torch.quasirandom.SobolEngine(dimension, scramble=False, seed=None)
## 四、Serialization [en](https://pytorch.org/docs/stable/torch.html#serialization)/[cn](https://pytorch.apachecn.org/docs/1.2/torch.html#serialization-%E5%BA%8F%E5%88%97%E5%8C%96)
* torch.save(obj, f, pickle_module=<module 'pickle' from '/scratch/rzou/pt/1.3-docs-env/lib/python3.7/pickle.py'>, pickle_protocol=2)
* torch.load(f, map_location=None, pickle_module=<module 'pickle' from '/scratch/rzou/pt/1.3-docs-env/lib/python3.7/pickle.py'>, **pickle_load_args)
## 五、Parallelism [en](https://pytorch.org/docs/stable/torch.html#parallelism)/[cn](https://pytorch.apachecn.org/docs/1.2/torch.html#parallelism-%E5%B9%B6%E8%A1%8C%E5%8C%96)
* torch.get_num_threads() → int
* torch.set_num_threads(int)
* torch.get_num_interop_threads() → int
* torch.set_num_interop_threads(int)
## 六、Locally disabling gradient computation [en](https://pytorch.org/docs/stable/torch.html#locally-disabling-gradient-computation)/cn
## 七、Math operations [en](https://pytorch.org/docs/stable/torch.html#math-operations)/[cn](https://pytorch.apachecn.org/docs/1.2/torch.html#math-operations-%E6%95%B0%E5%AD%A6%E6%93%8D%E4%BD%9C)
### 1、Pointwise Ops
* torch.abs(input, out=None) → Tensor
* torch.acos(input, out=None) → Tensor
* torch.add()
* torch.addcdiv(input, value=1, tensor1, tensor2, out=None) → Tensor
* torch.addcmul(input, value=1, tensor1, tensor2, out=None) → Tensor
* torch.asin(input, out=None) → Tensor
* torch.atan(input, out=None) → Tensor
* torch.atan2(input, other, out=None) → Tensor
* torch.bitwise_not(input, out=None) → Tensor
* torch.ceil(input, out=None) → Tensor
* torch.clamp(input, min, max, out=None) → Tensor
* torch.cos(input, out=None) → Tensor
* torch.cosh(input, out=None) → Tensor
* torch.div()
* torch.digamma(input, out=None) → Tensor
* torch.erf(input, out=None) → Tensor
* torch.erfc(input, out=None) → Tensor
* torch.erfinv(input, out=None) → Tensor
* torch.exp(input, out=None) → Tensor
* torch.expm1(input, out=None) → Tensor
* torch.floor(input, out=None) → Tensor
* torch.fmod(input, other, out=None) → Tensor
* torch.frac(input, out=None) → Tensor
* torch.lerp(input, end, weight, out=None)
* torch.log(input, out=None) → Tensor
* torch.log10(input, out=None) → Tensor
* torch.log1p(input, out=None) → Tensor
* torch.log2(input, out=None) → Tensor
* torch.logical_not(input, out=None) → Tensor
* torch.logical_xor(input, other, out=None) → Tensor
* torch.mul()
* torch.mvlgamma(input, p) → Tensor
* torch.neg(input, out=None) → Tensor
* torch.pow()
* torch.reciprocal(input, out=None) → Tensor
* torch.remainder(input, other, out=None) → Tensor
* torch.round(input, out=None) → Tensor
* torch.rsqrt(input, out=None) → Tensor
* torch.sigmoid(input, out=None) → Tensor
* torch.sign(input, out=None) → Tensor
* torch.sin(input, out=None) → Tensor
* torch.sinh(input, out=None) → Tensor
* torch.sqrt(input, out=None) → Tensor
* torch.tan(input, out=None) → Tensor
* torch.tanh(input, out=None) → Tensor
* torch.trunc(input, out=None) → Tensor
### 2、Reduction Ops
* torch.argmax()
* torch.argmin()
* torch.cumprod(input, dim, out=None, dtype=None) → Tensor
* torch.cumsum(input, dim, out=None, dtype=None) → Tensor
* torch.dist(input, other, p=2) → Tensor
* torch.logsumexp(input, dim, keepdim=False, out=None)
* torch.mean()
* torch.median()
* torch.mode(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)
* torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)
* torch.prod()
* torch.std()
* torch.std_mean()
* torch.sum()
* torch.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)
* torch.unique_consecutive(input, return_inverse=False, return_counts=False, dim=None)
* torch.var()
* torch.var_mean()
### 3、Comparison Ops
* torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) → bool
* torch.argsort(input, dim=-1, descending=False, out=None) → LongTensor
* torch.eq(input, other, out=None) → Tensor
* torch.equal(input, other) → bool
* torch.ge(input, other, out=None) → Tensor
* torch.gt(input, other, out=None) → Tensor
* torch.isfinite(tensor)
* torch.isinf(tensor)
* torch.isnan()
* torch.kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)
* torch.le(input, other, out=None) → Tensor
* torch.lt(input, other, out=None) → Tensor
* torch.max()
* torch.min()
* torch.ne(input, other, out=None) → Tensor
* torch.sort(input, dim=-1, descending=False, out=None) -> (Tensor, LongTensor)
* torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
### 4、Spectral Ops
* torch.fft(input, signal_ndim, normalized=False) → Tensor
* torch.ifft(input, signal_ndim, normalized=False) → Tensor
* torch.rfft(input, signal_ndim, normalized=False, onesided=True) → Tensor
* torch.irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None) → Tensor
* torch.stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True)
* torch.bartlett_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.blackman_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
* torch.hann_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
### 5、Other Operations
* torch.bincount(input, weights=None, minlength=0) → Tensor
* torch.broadcast_tensors(*tensors) → List of Tensors
* torch.cartesian_prod(*tensors)
* torch.cdist(x1, x2, p=2) → Tensor
* torch.combinations(input, r=2, with_replacement=False) → seq
* torch.cross(input, other, dim=-1, out=None) → Tensor
* torch.diag(input, diagonal=0, out=None) → Tensor
* torch.diag_embed(input, offset=0, dim1=-2, dim2=-1) → Tensor
* torch.diagflat(input, offset=0) → Tensor
* torch.diagonal(input, offset=0, dim1=0, dim2=1) → Tensor
* torch.einsum(equation, *operands) → Tensor
* torch.flatten(input, start_dim=0, end_dim=-1) → Tensor
* torch.flip(input, dims) → Tensor
* torch.rot90(input, k, dims) → Tensor
* torch.histc(input, bins=100, min=0, max=0, out=None) → Tensor
* torch.meshgrid(*tensors, **kwargs)
* torch.renorm(input, p, dim, maxnorm, out=None) → Tensor
* torch.repeat_interleave()
* torch.roll(input, shifts, dims=None) → Tensor
* torch.tensordot(a, b, dims=2)
* torch.trace(input) → Tensor
* torch.tril(input, diagonal=0, out=None) → Tensor
* torch.tril_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) → Tensor
* torch.triu(input, diagonal=0, out=None) → Tensor
* torch.triu_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) → Tensor
### 6、BLAS and LAPACK Operations
* torch.addbmm(beta=1, input, alpha=1, batch1, batch2, out=None) → Tensor
* torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None) → Tensor
* torch.addmv(beta=1, input, alpha=1, mat, vec, out=None) → Tensor
* torch.addr(beta=1, input, alpha=1, vec1, vec2, out=None) → Tensor
* torch.baddbmm(beta=1, input, alpha=1, batch1, batch2, out=None) → Tensor
* torch.bmm(input, mat2, out=None) → Tensor
* torch.chain_matmul(*matrices)
* torch.cholesky(input, upper=False, out=None) → Tensor
* torch.cholesky_inverse(input, upper=False, out=None) → Tensor
* torch.cholesky_solve(input, input2, upper=False, out=None) → Tensor
* torch.dot(input, tensor) → Tensor
* torch.eig(input, eigenvectors=False, out=None) -> (Tensor, Tensor)
* torch.geqrf(input, out=None) -> (Tensor, Tensor)
* torch.ger(input, vec2, out=None) → Tensor
* torch.inverse(input, out=None) → Tensor 
* torch.det(input) → Tensor
* torch.logdet(input) → Tensor
* torch.slogdet(input) -> (Tensor, Tensor)
* torch.lstsq(input, A, out=None) → Tensor
* torch.lu(A, pivot=True, get_infos=False, out=None)
* torch.lu_solve(input, LU_data, LU_pivots, out=None) → Tensor
* torch.lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True)
* torch.matmul(input, other, out=None) → Tensor
* torch.matrix_power(input, n) → Tensor
* torch.matrix_rank(input, tol=None, bool symmetric=False) → Tensor
* torch.mm(input, mat2, out=None) → Tensor
* torch.mv(input, vec, out=None) → Tensor
* torch.orgqr(input, input2) → Tensor
* torch.ormqr(input, input2, input3, left=True, transpose=False) → Tensor
* torch.pinverse(input, rcond=1e-15) → Tensor
* torch.qr(input, some=True, out=None) -> (Tensor, Tensor)
* torch.solve(input, A, out=None) -> (Tensor, Tensor)
* torch.svd(input, some=True, compute_uv=True, out=None) -> (Tensor, Tensor, Tensor)
* torch.symeig(input, eigenvectors=False, upper=True, out=None) -> (Tensor, Tensor)
* torch.trapz()
* torch.triangular_solve(input, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)
## 八、Utilities [en](https://pytorch.org/docs/stable/torch.html#utilities)/cn
* torch.compiled_with_cxx11_abi()
* torch.result_type(tensor1, tensor2) → dtype
* torch.can_cast(from, to) → bool
* torch.promote_types(type1, type2) → dtype

