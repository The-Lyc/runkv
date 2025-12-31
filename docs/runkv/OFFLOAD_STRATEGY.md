# llama.cpp Offload 策略详解

本文档详细说明 llama.cpp 如何在运行时应用 offload 策略，将模型的不同部分分配到不同的计算设备（CPU/GPU）上。

## 一、Offload 配置参数

### 1. 模型参数 (`llama_model_params`)

```cpp
struct llama_model_params {
    // 设备列表
    ggml_backend_dev_t * devices;                                // NULL = 使用所有可用设备
    
    // Tensor buffer类型覆盖
    const llama_model_tensor_buft_override * tensor_buft_overrides;
    
    // GPU层数控制
    int32_t n_gpu_layers;                                        // 卸载到 VRAM 的层数，-1 表示全部
    enum llama_split_mode split_mode;                            // 如何跨多个 GPU 分割模型
    
    // GPU分割控制
    int32_t main_gpu;                                            // 主 GPU（split_mode=NONE时使用）
    const float * tensor_split;                                  // 各GPU的比例（按层或按行）
    
    ...
};
```

### 2. 上下文参数 (`llama_context_params`)

```cpp
struct llama_context_params {
    bool offload_kqv;     // 将 KQV 操作（包括 KV cache）卸载到 GPU
    bool op_offload;      // 将主机 tensor 操作卸载到设备
    ...
};
```

## 二、Offload 策略的三个层次

### Layer 1: 模型权重的设备分配（加载时）

在模型加载时完成，决定每个权重张量存储在哪个设备上。

#### 1.1 设备初始化

```cpp
// src/llama.cpp: llama_model_load_from_file_impl()
// 构建可用设备列表
for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_GPU:      // CUDA/Metal/Vulkan等
        case GGML_BACKEND_DEVICE_TYPE_GPU_FULL: // 专用GPU
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:    // 加速器
        // 添加到 model->devices
    }
}
```

#### 1.2 按层分配设备

```cpp
// src/llama-model.cpp
pimpl->dev_layer.resize(n_layer);
for (uint32_t il = 0; il < n_layer; il++) {
    pimpl->dev_layer[il] = get_layer_buft_list(il);
}

// get_layer_buft_list() 根据 n_gpu_layers 决定设备：
//   - il < (n_layer - n_gpu_layers): CPU
//   - il >= (n_layer - n_gpu_layers): GPU
```

**分配策略**：
- **n_gpu_layers = 0**: 所有层在 CPU
- **n_gpu_layers = 10**: 后 10 层在 GPU，其余在 CPU
- **n_gpu_layers = -1**: 所有层在 GPU

#### 1.3 Tensor Buffer 类型分配

```cpp
// src/llama-model-loader.cpp
// 为每个 tensor 选择 buffer type
auto * buft = get_buft_list(tn.idx, tensor_name);

// 根据层ID和tensor类型选择：
if (il < gpu_layer_start) {
    buft = CPU_buffer_type;
} else {
    buft = GPU_buffer_type;
}
```

**Tensor 分类**：
- **Input tensors** (token_embd, pos_embd): 通常在 CPU
- **Layer weights** (attn_q, attn_k, ffn_gate等): 根据层ID分配
- **Output tensors** (output_norm, output): 根据 n_gpu_layers 分配

#### 1.4 实际加载权重

```cpp
// src/llama-model-loader.cpp: load_all_data()
// 为每个 buffer type 分配内存
for (auto & [buft, buffers] : bufs_grouped) {
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
}

// 从文件读取权重数据到对应的 buffer
for (auto & tensor : tensors) {
    ggml_backend_tensor_set(tensor, data, 0, size);
}
```

### Layer 2: KV Cache 的设备分配（上下文创建时）

KV cache 的分配策略由 `offload_kqv` 参数控制。

```cpp
// src/llama-kv-cache.cpp: llama_kv_cache 构造函数
for (uint32_t il = 0; il < n_layer; il++) {
    ggml_backend_buffer_type_t buft;
    
    if (offload) {
        // 获取该层权重所在的设备
        auto * dev = model.dev_layer(il);
        buft = ggml_backend_dev_buffer_type(dev);
    } else {
        // 保持在 CPU
        buft = ggml_backend_cpu_buffer_type();
    }
    
    // 在对应设备上创建 K/V tensors
    ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k, kv_size, n_stream);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v, kv_size, n_stream);
}
```

**分配结果**：
- `offload_kqv = false`: 所有 KV cache 在 CPU 内存
- `offload_kqv = true`: KV cache 跟随对应层的权重设备
  - 第 0-9 层在 CPU → KV cache 在 CPU
  - 第 10-31 层在 GPU → KV cache 在 GPU VRAM

### Layer 3: 计算图操作的设备分配（推理时）

在每次推理时，Backend Scheduler 动态分配每个操作到合适的设备。

#### 3.1 Backend Scheduler 初始化

```cpp
// src/llama-context.cpp: llama_context 构造函数
// 创建 backend 列表
std::vector<ggml_backend_t> backends;
backends.push_back(gpu_backend);   // GPU backend (CUDA/Metal/...)
backends.push_back(cpu_backend);   // CPU backend

// 创建 scheduler
sched = ggml_backend_sched_new(
    backends.data(),
    buffer_types.data(),
    n_backends,
    max_nodes,
    pipeline_parallel,
    cparams.op_offload     // ← 启用 op_offload
);
```

#### 3.2 计算图构建

```cpp
// src/llama-graph.cpp: llm_build_*()
// 构建 transformer 计算图
ggml_cgraph * gf = model.build_graph(gparams);

// 例如：attention 操作
cur = ggml_mul_mat(ctx0, k_cache, q);     // Q·K^T
cur = ggml_soft_max(ctx0, cur);           // softmax
cur = ggml_mul_mat(ctx0, v_cache, cur);   // attn·V
```

#### 3.3 Backend 自动分配（Pass 1-4）

Backend Scheduler 通过多个 pass 自动分配每个操作的执行设备：

```cpp
// ggml/src/ggml-backend.cpp: ggml_backend_sched_split_graph()

// Pass 1: 根据输入张量位置分配 backend
for (node : graph->nodes) {
    int backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
    // 规则：
    // - 如果输入在 GPU weights buffer → 分配到 GPU
    // - 如果操作支持 offload → 分配到 GPU
    // - 否则 → CPU
}

// Pass 2: 扩展 backend 分配到相邻节点
// 将相同 backend 的操作合并，减少数据传输

// Pass 3: 升级节点到更高优先级 backend
// 如果节点的输入已经在 GPU 且 GPU 支持该操作，升级到 GPU

// Pass 4: 为剩余节点分配 backend 并处理数据复制
for (node : unassigned_nodes) {
    // 插入必要的数据复制操作
    if (input_backend != node_backend) {
        insert_copy_node(input, node_backend);
    }
}
```

**分配规则示例**：

| 操作 | 输入位置 | 支持的 Backend | 最终分配 |
|------|---------|---------------|---------|
| `token_embd` (GET_ROWS) | Weights:CPU | CPU, GPU | CPU (输入在 CPU) |
| `attn_q` (MUL_MAT) | Weights:GPU | CPU, GPU | GPU (权重在 GPU) |
| `rope` | Input:GPU | CPU, GPU | GPU (输入在 GPU) |
| `soft_max` | Input:GPU | CPU, GPU | GPU (输入在 GPU) |
| `add` | Input:GPU | CPU, GPU | GPU (输入在 GPU) |

#### 3.4 Op Offload 机制

`op_offload` 参数启用更激进的卸载策略：

```cpp
// ggml/src/ggml-backend.cpp
if (op_offload && ggml_backend_offload_op(gpu_backend, node)) {
    // 即使权重在 CPU，如果操作昂贵（如大矩阵乘法）
    // 也将其卸载到 GPU 执行
    backend_id = gpu_backend_id;
}
```

**适用操作**：
- `MUL_MAT`, `MUL_MAT_ID`: 大矩阵乘法
- `CONV_TRANSPOSE_1D`: 卷积操作
- 计算密集型操作

#### 3.5 分割图为子图

Scheduler 将计算图分割为多个子图，每个子图在单一 backend 上执行：

```cpp
// ggml/src/ggml-backend.cpp: ggml_backend_sched_split_graph()
// 分割规则：
// - 当 backend 切换时创建新的 split
// - 插入跨 backend 的数据复制节点

splits[0]: backend=GPU, nodes=[0-15]     // GPU 操作
  copy: GPU→CPU                          // 数据传输
splits[1]: backend=CPU, nodes=[16-20]    // CPU 操作
  copy: CPU→GPU                          // 数据传输
splits[2]: backend=GPU, nodes=[21-50]    // GPU 操作
```

#### 3.6 执行计算

```cpp
// src/llama-context.cpp: graph_compute()
ggml_backend_sched_graph_compute_async(sched, gf);

// 内部流程：
// 1. 为每个 split 分配 compute buffer
// 2. 复制输入数据到目标 backend
// 3. 在对应 backend 上执行子图
// 4. 复制输出数据（如果需要）
```

## 三、完整推理流程中的 Offload 应用

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. 模型加载阶段                                │
│  根据 n_gpu_layers 分配权重到 CPU/GPU                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Layer 0-19   │  │ Layer 20-31  │  │ Output Layer │          │
│  │   Weights    │  │   Weights    │  │   Weights    │          │
│  │   [CPU]      │  │   [GPU]      │  │   [GPU]      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  2. 上下文创建阶段                                 │
│  根据 offload_kqv 分配 KV Cache                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ KV Cache     │  │ KV Cache     │                             │
│  │ Layer 0-19   │  │ Layer 20-31  │                             │
│  │   [CPU]      │  │   [GPU]      │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   3. 推理执行阶段                                 │
│  Backend Scheduler 动态分配操作                                  │
│                                                                   │
│  Input Tokens [CPU]                                              │
│       ↓                                                           │
│  Token Embedding (GET_ROWS) [CPU]  ← Weights 在 CPU             │
│       ↓                                                           │
│  Position Embedding [CPU]                                        │
│       ↓ [copy CPU→GPU]                                           │
│  ┌─────────────────────────────────────┐                        │
│  │     Layer 0-19 (CPU Weights)        │                        │
│  │  · Attention Q/K/V [GPU] ← op_offload 激活                   │
│  │  · Rope [GPU]                         │                        │
│  │  · Attention [CPU] ← KV cache 在 CPU │                        │
│  │  · FFN [GPU] ← op_offload 激活        │                        │
│  └─────────────────────────────────────┘                        │
│       ↓                                                           │
│  ┌─────────────────────────────────────┐                        │
│  │     Layer 20-31 (GPU Weights)       │                        │
│  │  · Attention Q/K/V [GPU]             │                        │
│  │  · Rope [GPU]                         │                        │
│  │  · Attention [GPU] ← KV cache 在 GPU │                        │
│  │  · FFN [GPU]                          │                        │
│  └─────────────────────────────────────┘                        │
│       ↓                                                           │
│  Output Norm [GPU]                                               │
│       ↓                                                           │
│  LM Head (MUL_MAT) [GPU]                                         │
│       ↓ [copy GPU→CPU]                                           │
│  Logits [CPU]                                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 四、关键数据流

### 示例：n_gpu_layers=20, offload_kqv=true

```
Token → [CPU]
  ↓
Token Embedding → [CPU] (weights 在 CPU)
  ↓ [copy→GPU]
Layer 0 Q/K/V Proj → [GPU] (op_offload)
  ↓ [copy→CPU]
Layer 0 Attention → [CPU] (KV cache 在 CPU)
  ↓ [copy→GPU]
Layer 0 FFN → [GPU] (op_offload)
  ↓ [copy→CPU]
...
Layer 19 → [CPU/GPU混合]
  ↓ [copy→GPU]
Layer 20 Q/K/V Proj → [GPU] (weights 在 GPU)
  ↓
Layer 20 Attention → [GPU] (KV cache 在 GPU)
  ↓
Layer 20 FFN → [GPU] (weights 在 GPU)
  ↓
...
Layer 31 → [GPU]
  ↓
Output → [GPU]
  ↓ [copy→CPU]
Logits → [CPU]
```

## 五、性能优化建议

### 1. 减少数据传输

**问题**: CPU-GPU 频繁数据传输成为瓶颈

**优化**:
- 设置 `n_gpu_layers` 为连续的层（避免交错）
- 如果内存充足，设置 `offload_kqv=true`
- 启用 `op_offload` 让更多操作在 GPU 执行

### 2. 平衡内存使用

**VRAM 不足时**:
```cpp
// 减少 GPU 层数
params.n_gpu_layers = 10;           // 只卸载 10 层
params.offload_kqv = false;          // KV cache 留在 CPU
```

**VRAM 充足时**:
```cpp
// 最大化 GPU 使用
params.n_gpu_layers = -1;            // 所有层到 GPU
params.offload_kqv = true;           // KV cache 也到 GPU
params.op_offload = true;            // 激进的操作卸载
```

### 3. 多 GPU 分割

**Row Split** (`split_mode = LLAMA_SPLIT_MODE_ROW`):
```cpp
// 按行分割大矩阵
params.split_mode = LLAMA_SPLIT_MODE_ROW;
params.tensor_split = {0.6, 0.4};    // GPU0:60%, GPU1:40%
```

**Layer Split** (`split_mode = LLAMA_SPLIT_MODE_LAYER`):
```cpp
// 按层分割
params.split_mode = LLAMA_SPLIT_MODE_LAYER;
// Layer 0-15 → GPU0, Layer 16-31 → GPU1
```

## 六、调试 Offload 行为

### 查看分配结果

```cpp
// 1. 启用日志
setenv("LLAMA_LOG_LEVEL", "DEBUG", 1);

// 2. 查看层分配
for (int il = 0; il < n_layer; il++) {
    auto * dev = model.dev_layer(il);
    printf("Layer %d: %s\n", il, ggml_backend_dev_name(dev));
}

// 3. 查看 backend scheduler 的分割
int n_splits = ggml_backend_sched_get_n_splits(sched);
printf("Graph splits: %d\n", n_splits);
```

### GDB 断点建议

```gdb
# 权重加载
break llama_model_loader::load_all_data

# 设备分配
break llama_model::dev_layer

# Backend 分配
break ggml_backend_sched_split_graph
break ggml_backend_sched_backend_id_from_cur

# 数据传输
break ggml_backend_tensor_copy
```

## 七、总结

llama.cpp 的 offload 策略是一个**三层机制**：

1. **静态层（模型加载）**: 根据 `n_gpu_layers` 决定权重存储位置
2. **准静态层（上下文创建）**: 根据 `offload_kqv` 决定 KV cache 位置
3. **动态层（推理执行）**: Backend Scheduler 根据数据位置和操作类型动态分配

这种设计在 **灵活性** 和 **性能** 之间取得了平衡：
- 用户通过简单参数控制整体策略
- 系统自动优化具体执行细节
- 支持 CPU/GPU/多GPU 等复杂配置
