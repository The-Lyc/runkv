# llama.cpp 动态 Offload 可行性分析

## 问题：能否在运行时动态 offload weight 和 KV cache？

**简短回答**：llama.cpp 当前架构**不支持运行时动态 offload**。权重和 KV cache 的设备分配在加载/创建时就固定了。

## 一、当前架构的限制

### 1. Weight Offload（模型权重）

**分配时机**：模型加载时（`llama_model_load_from_file()`）

```cpp
// src/llama-model.cpp
// 权重分配是一次性的
for (uint32_t il = 0; il < n_layer; il++) {
    pimpl->dev_layer[il] = get_layer_buft_list(il);  // 固定设备
}

// src/llama-model-loader.cpp
// 权重加载到固定的 buffer
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
ggml_backend_tensor_set(tensor, data, 0, size);  // 数据写入固定位置
```

**限制原因**：
1. **Buffer 生命周期**：权重存储在 `ggml_backend_buffer` 中，这些 buffer 在模型加载时创建，在模型释放时销毁
2. **Tensor 元数据固定**：每个 tensor 的 `buffer` 指针在创建后就固定，指向特定设备的内存
3. **无迁移 API**：ggml backend 层没有提供 tensor 跨设备迁移的 API

### 2. KV Cache Offload

**分配时机**：上下文创建时（`llama_init_from_model()`）

```cpp
// src/llama-kv-cache.cpp: llama_kv_cache 构造函数
for (uint32_t il = 0; il < n_layer; il++) {
    if (offload) {
        auto * dev = model.dev_layer(il);
        buft = ggml_backend_dev_buffer_type(dev);  // 跟随权重设备
    } else {
        buft = ggml_backend_cpu_buffer_type();     // 固定在 CPU
    }
    
    // 在固定设备上创建 K/V tensors
    ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, ...);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, ...);
}
```

**限制原因**：
1. **静态分配**：KV cache 大小和位置在上下文创建时确定，之后不变
2. **绑定到 context**：KV cache 是 `llama_context` 的一部分，context 的生命周期内不能改变
3. **依赖权重位置**：KV cache 的设备分配通常跟随对应层权重的位置

## 二、为什么不能动态迁移？

### 技术障碍

1. **指针固定性**
```cpp
struct ggml_tensor {
    void * data;                    // 指向设备内存的指针，固定不变
    ggml_backend_buffer_t buffer;   // 所属 buffer，固定不变
    ...
};
```

如果要迁移 tensor：
- 需要分配新的 buffer
- 复制数据
- 更新所有引用该 tensor 的地方
- 释放旧 buffer

2. **计算图依赖**
```cpp
// 计算图中的操作依赖 tensor 位置
ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
// output 的 backend 分配依赖 weight 的位置
// 如果 weight 移动，需要重新构建整个图
```

3. **Backend Scheduler 的假设**
```cpp
// ggml/src/ggml-backend.cpp: ggml_backend_sched_split_graph()
// Scheduler 假设 tensor 位置在图构建后不变
if (tensor->buffer != NULL) {
    // 预分配的 tensor 不能移动到其他 backend
    int backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
}
```

### 性能考虑

即使实现了动态迁移，也会有严重的性能问题：

```
迁移开销 = 数据传输时间 + 图重建时间 + 同步开销

以 7B 模型为例（FP16）：
- 单层权重: ~500 MB
- PCIe 3.0 传输: ~500 MB / 12 GB/s ≈ 42ms
- 迁移一层的最小开销: >50ms
- 生成一个 token: ~20-50ms

结论：迁移开销 >> 计算时间，完全不划算
```

## 三、当前的"动态"机制

虽然不能动态迁移权重/KV cache，但有以下动态机制：

### 1. 操作级别的动态 Offload（`op_offload`）

这是**唯一的运行时动态决策**，但只针对**计算操作**，不针对数据位置。

```cpp
// ggml/src/ggml-backend.cpp
if (sched->op_offload && 
    src_backend_id == sched->n_backends - 1 &&  // 权重在 CPU
    ggml_backend_buffer_is_host(src->buffer)) {
    
    for (int b = 0; b < src_backend_id; b++) {
        // 检查 GPU backend 是否想要执行这个操作
        if (ggml_backend_offload_op(sched->backends[b], tensor)) {
            // 将操作分配到 GPU，即使权重在 CPU
            // 框架会自动插入数据传输操作
            return b;
        }
    }
}
```

**工作原理**：
- 权重仍在 CPU
- 每次计算前：CPU → GPU 传输权重
- 在 GPU 执行昂贵操作（如矩阵乘法）
- 计算后：GPU → CPU 传输结果

**适用场景**：
```cpp
// ggml/src/ggml-cuda/ggml-cuda.cu
static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;
    return get_op_batch_size(op) >= min_batch_size;  // 只对大 batch 有效
}
```

### 2. 计算图级别的 Backend 重分配

每个 batch 的计算图可以有不同的 backend 分配：

```cpp
// src/llama-context.cpp
for (each batch) {
    // 根据 batch 特征构建图
    auto * gf = model.build_graph(gparams);
    
    // 重新分配 backend（但 tensor 位置不变）
    ggml_backend_sched_split_graph(sched.get(), gf);
    
    // 执行
    ggml_backend_sched_graph_compute_async(sched.get(), gf);
}
```

这允许：
- Prefill 阶段（大 batch）：更多操作在 GPU
- Decode 阶段（小 batch）：更多操作在 CPU

但**数据位置始终不变**。

## 四、可能的解决方案（需要架构修改）

### 方案 1: 多 Context 策略

**思路**：为不同场景创建多个 context

```cpp
// 场景 A: 低延迟（KV cache 在 GPU）
llama_context_params params_gpu;
params_gpu.offload_kqv = true;
llama_context * ctx_gpu = llama_init_from_model(model, params_gpu);

// 场景 B: 低内存（KV cache 在 CPU）
llama_context_params params_cpu;
params_cpu.offload_kqv = false;
llama_context * ctx_cpu = llama_init_from_model(model, params_cpu);

// 运行时切换
if (need_low_latency) {
    llama_decode(ctx_gpu, batch);
} else {
    llama_decode(ctx_cpu, batch);
}
```

**限制**：
- ❌ 需要维护多个 context（内存开销大）
- ❌ 不能在同一个序列中切换（KV cache 不共享）
- ✓ 可以为不同请求使用不同策略

### 方案 2: 重新加载模型

**思路**：需要改变策略时，释放并重新加载

```cpp
// 初始配置
llama_model_params mparams = {...};
mparams.n_gpu_layers = 20;
llama_model * model = llama_model_load_from_file(path, mparams);
llama_context * ctx = llama_init_from_model(model, cparams);

// ... 使用一段时间 ...

// 需要改变策略
llama_free(ctx);
llama_model_free(model);

// 重新加载
mparams.n_gpu_layers = 30;  // 改变配置
model = llama_model_load_from_file(path, mparams);
ctx = llama_init_from_model(model, cparams);
```

**限制**：
- ❌ 重新加载时间长（几秒到几十秒）
- ❌ 所有推理状态丢失
- ✓ 完全改变配置

### 方案 3: KV Cache 部分迁移（理论方案）

**思路**：实现 KV cache 的跨设备复制

```cpp
// 伪代码 - 当前 llama.cpp 不支持
bool llama_kv_cache_migrate(llama_context * ctx, int layer_id, ggml_backend_t new_backend) {
    // 1. 在新设备上分配空间
    auto * new_k = allocate_on_backend(new_backend, ...);
    auto * new_v = allocate_on_backend(new_backend, ...);
    
    // 2. 复制数据
    copy_tensor(old_k, new_k);
    copy_tensor(old_v, new_v);
    
    // 3. 更新引用
    ctx->kv_cache.layers[layer_id].k = new_k;
    ctx->kv_cache.layers[layer_id].v = new_v;
    
    // 4. 释放旧空间
    free_tensor(old_k);
    free_tensor(old_v);
}
```

**需要的修改**：
1. 实现 `ggml_tensor_migrate()` API
2. 修改 Backend Scheduler 支持动态 tensor 位置
3. 处理计算图重建
4. 处理并发和同步问题

**估计工作量**：数千行代码，需要重构核心架构

## 五、实用建议

### 1. 静态配置最优化

在启动时就选择最佳配置：

```cpp
// 测量可用内存
size_t gpu_free, gpu_total;
ggml_backend_dev_memory(gpu_dev, &gpu_free, &gpu_total);

// 根据内存自动设置 n_gpu_layers
llama_model_params mparams = llama_model_default_params();
mparams.n_gpu_layers = estimate_optimal_gpu_layers(gpu_free, model_size);

llama_context_params cparams = llama_context_default_params();
cparams.offload_kqv = (gpu_free > kv_cache_size + safety_margin);
```

### 2. 使用自动调整工具

```bash
# llama.cpp 提供的自动拟合工具
./llama-cli --model model.gguf --params-fit
# 会自动计算最优的 n_gpu_layers 和 n_ctx
```

### 3. 分层策略

针对不同场景使用不同进程：

```bash
# 进程 1: 高性能推理（高 VRAM 使用）
./llama-server --model model.gguf -ngl -1 --offload-kqv

# 进程 2: 高吞吐量推理（低 VRAM 使用）
./llama-server --model model.gguf -ngl 20 --no-offload-kqv --port 8081
```

### 4. 监控和预警

```cpp
// 监控内存使用
void monitor_memory() {
    size_t free, total;
    ggml_backend_dev_memory(dev, &free, &total);
    
    if (free < threshold) {
        // 警告：即将 OOM
        // 可以拒绝新请求或清理旧 KV cache
        llama_memory_seq_rm(mem, old_seq_id, -1, -1);
    }
}
```

## 六、总结

| 特性 | 是否支持 | 原因 |
|------|---------|------|
| 动态 Weight Offload | ❌ 不支持 | Buffer 固定，无迁移 API |
| 动态 KV Cache Offload | ❌ 不支持 | 绑定到 context，无迁移机制 |
| 动态操作 Offload | ✅ 支持 | `op_offload` 参数启用 |
| 重新加载改变配置 | ✅ 可行 | 但开销大，丢失状态 |
| 多 Context 策略 | ✅ 可行 | 内存开销大 |

**核心结论**：
1. llama.cpp 的当前架构**不支持运行时动态迁移权重和 KV cache**
2. 设备分配是**静态的**，在加载/创建时确定
3. 唯一的"动态"是 **op_offload**，但只影响计算位置，不影响数据位置
4. 如果需要改变配置，只能**重新加载模型/创建 context**

**推荐做法**：
- 启动时就选择最优配置
- 使用 `--params-fit` 自动计算参数
- 监控内存使用，提前预警
- 多进程/多 context 处理不同场景

实现真正的动态 offload 需要对 llama.cpp 进行**重大架构修改**，包括：
- Tensor 迁移 API
- 动态图重建
- Buffer 生命周期管理
- Backend Scheduler 重构

这将是一个**数月的工程工作**，目前没有看到社区有这样的计划。
