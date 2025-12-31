# KV Cache 按 Layer 存储示例

本文档通过实际代码展示 llama.cpp 中 KV cache 如何按 layer 存储，以及 `cpy_k` 和 `get_k` 的使用方式。

## 1. 数据结构回顾

### 每层独立的 K/V tensor
```cpp
// 位置: src/llama-kv-cache.h:206-216
struct kv_layer {
    uint32_t il;  // 模型中的 layer 索引
    
    ggml_tensor * k;  // K cache: [n_embd_k_gqa, kv_size, n_stream]
    ggml_tensor * v;  // V cache: [n_embd_v_gqa, kv_size, n_stream]
    
    std::vector<ggml_tensor *> k_stream;  // 每个 stream 的视图
    std::vector<ggml_tensor *> v_stream;
};

std::vector<kv_layer> layers;  // 关键：每层一个独立的 tensor
```

**关键点：**
- `layers[0].k` 存储第 0 层的所有 K 值
- `layers[1].k` 存储第 1 层的所有 K 值
- ...依此类推，**按 layer 分离存储**

## 2. get_k() - 读取某层的 K cache

### 函数签名与实现
```cpp
// 位置: src/llama-kv-cache.cpp:1008-1027
ggml_tensor * llama_kv_cache::get_k(
    ggml_context * ctx,
    int32_t il,              // 输入：layer 索引
    uint32_t n_kv,           // 要读取的 cell 数量
    const slot_info & sinfo  // cell 映射信息
) const {
    // 步骤 1: 通过 layer 索引找到对应的 kv_layer
    const int32_t ikv = map_layer_ids.at(il);
    
    // 步骤 2: 获取该层的 K tensor
    auto * k = layers[ikv].k;  // 这是第 il 层的 K cache
    
    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];
    
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    
    // 步骤 3: 创建一个 4D 视图用于 attention 计算
    // 返回形状: [n_embd_head_k, n_head_kv, n_kv, n_stream]
    return ggml_view_4d(ctx, k,
        hparams.n_embd_head_k, hparams.n_head_kv(il), n_kv, ns,
        ggml_row_size(k->type, hparams.n_embd_head_k),      // stride for heads
        ggml_row_size(k->type, n_embd_k_gqa),               // stride for tokens
        ggml_row_size(k->type, n_embd_k_gqa*kv_size),       // stride for streams
        ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}
```

**体现按 layer 存储的关键：**
```cpp
auto * k = layers[ikv].k;  // 直接索引到特定层的 tensor
```

### 在实际 attention 计算中的使用
```cpp
// 位置: src/llama-graph.cpp:1676
// 在构建第 il 层的 attention 图时

ggml_tensor * k = mctx_cur->get_k(ctx0, il);  // 获取第 il 层的 K
ggml_tensor * v = mctx_cur->get_v(ctx0, il);  // 获取第 il 层的 V

// k 现在包含第 il 层的所有 K 值，形状: [head_dim, n_heads, n_kv, n_stream]
// 用于计算 QK^T
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);  // Q @ K^T
```

## 3. cpy_k() - 写入某层的 K cache

### 函数签名与实现
```cpp
// 位置: src/llama-kv-cache.cpp:1060-1093
ggml_tensor * llama_kv_cache::cpy_k(
    ggml_context * ctx,
    ggml_tensor * k_cur,     // 输入：当前计算的 K [n_embd_head_k, n_head_k, n_tokens]
    ggml_tensor * k_idxs,    // 输入：cell 索引 [n_tokens]
    int32_t il,              // 输入：layer 索引
    const slot_info & sinfo
) const {
    // 步骤 1: 通过 layer 索引找到对应的 kv_layer
    const int32_t ikv = map_layer_ids.at(il);
    
    // 步骤 2: 获取该层的 K tensor（目标位置）
    ggml_tensor * k = layers[ikv].k;  // 这是第 il 层的 K cache
    
    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];
    
    const int64_t n_embd_gqa = n_embd_head*n_head;
    
    // 步骤 3: 将 k_cur 重塑为 2D: [n_embd_gqa, n_tokens]
    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);
    
    const int64_t n_stream = k->ne[2];
    
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        // 合并所有 streams 成一个大的 2D tensor
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }
    
    // 步骤 4: 使用 set_rows 按索引写入
    // k[k_idxs[i]] = k_cur[i] for each token i
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

**体现按 layer 存储的关键：**
```cpp
ggml_tensor * k = layers[ikv].k;  // 写入特定层的 tensor
```

### 在实际计算中的使用
```cpp
// 位置: src/llama-graph.cpp:1669
// 在第 il 层前向传播时

// 计算当前 batch 的 K 值
ggml_tensor * k_cur = ggml_mul_mat(ctx0, wk, cur);  // [head_dim, n_heads, n_tokens]

// 准备 cell 索引
const auto & k_idxs = inp->get_k_idxs();  // [n_tokens]，每个值是 cell 索引

// 将 k_cur 写入第 il 层的 K cache
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
```

## 4. 完整流程示例

假设处理一个 2 层模型，batch 包含 3 个 tokens：

### 初始化
```cpp
// 构造时创建每层的 tensor
layers[0].k = ggml_new_tensor_3d(ctx, type_k, 4096, 512, 1);  // 第 0 层
layers[1].k = ggml_new_tensor_3d(ctx, type_k, 4096, 512, 1);  // 第 1 层
```

### Layer 0 的处理
```cpp
// 1. 计算 Layer 0 的 K 值
ggml_tensor * k_cur_l0 = compute_k_layer_0(input);  // [128, 32, 3]
                                                     // [head_dim, n_heads, n_tokens]

// 2. 分配 cells (假设分配到 cell 5, 6, 7)
k_idxs->data = {5, 6, 7};

// 3. 写入 Layer 0 的 K cache
cpy_k(ctx, k_cur_l0, k_idxs, il=0);
// 执行: layers[0].k[..., 5] = k_cur_l0[..., 0]
//       layers[0].k[..., 6] = k_cur_l0[..., 1]
//       layers[0].k[..., 7] = k_cur_l0[..., 2]

// 4. 读取 Layer 0 的 K cache (假设已有 10 个 tokens)
ggml_tensor * k_l0 = get_k(ctx, il=0, n_kv=10);
// 返回: layers[0].k[..., 0:10]  // 只从 Layer 0 读取
```

### Layer 1 的处理
```cpp
// 1. 计算 Layer 1 的 K 值
ggml_tensor * k_cur_l1 = compute_k_layer_1(hidden);  // [128, 32, 3]

// 2. 使用相同的 cell 索引 {5, 6, 7}
k_idxs->data = {5, 6, 7};

// 3. 写入 Layer 1 的 K cache（注意：不同的 tensor）
cpy_k(ctx, k_cur_l1, k_idxs, il=1);
// 执行: layers[1].k[..., 5] = k_cur_l1[..., 0]  // 写入 Layer 1 的 tensor
//       layers[1].k[..., 6] = k_cur_l1[..., 1]
//       layers[1].k[..., 7] = k_cur_l1[..., 2]

// 4. 读取 Layer 1 的 K cache
ggml_tensor * k_l1 = get_k(ctx, il=1, n_kv=10);
// 返回: layers[1].k[..., 0:10]  // 只从 Layer 1 读取
```

## 5. 关键总结

### Cell 管理（token 维度）
- **全局共享**：所有 layer 使用相同的 cell 分配
- `v_cells` 记录哪些 cells 被占用，以及它们的 token positions
- Cell 索引（如 5, 6, 7）在所有 layer 中含义相同

### 数据存储（layer 维度）
- **按 layer 分离**：每层有独立的 `layers[il].k` 和 `layers[il].v` tensor
- Cell 5 在 Layer 0 和 Layer 1 中存储**不同的数据**：
  - `layers[0].k[..., 5]` 存储 Layer 0 的某个 token 的 K 值
  - `layers[1].k[..., 5]` 存储 Layer 1 的**同一个 token** 的 K 值

### 读写操作
```
写入 (cpy_k):
  token → cell 索引 → 在特定 layer 的 tensor 中写入
  
读取 (get_k):
  指定 layer → 返回该 layer tensor 的视图 → 用于该 layer 的 attention

关键：每次 cpy_k/get_k 都通过 il 参数指定 layer，
      从而操作 layers[il].k 这个**独立的** tensor
```

## 6. 内存布局可视化

```
KV Cache 内存结构:

layers[0].k:  [embedding_dim, kv_size, n_stream]
              ┌─────────────────────────────────┐
    Cell 0 →  │ Layer 0, Token at pos=0  的 K   │
    Cell 1 →  │ Layer 0, Token at pos=1  的 K   │
    Cell 2 →  │ Layer 0, Token at pos=2  的 K   │
              │            ...                  │
              └─────────────────────────────────┘

layers[1].k:  [embedding_dim, kv_size, n_stream]
              ┌─────────────────────────────────┐
    Cell 0 →  │ Layer 1, Token at pos=0  的 K   │
    Cell 1 →  │ Layer 1, Token at pos=1  的 K   │
    Cell 2 →  │ Layer 1, Token at pos=2  的 K   │
              │            ...                  │
              └─────────────────────────────────┘

v_cells:      统一管理所有 layer 的 cell 分配
              ┌────────┬──────┬────────────┐
    Cell 0 →  │ pos=0  │ used │ seq_id=0   │
    Cell 1 →  │ pos=1  │ used │ seq_id=0   │
    Cell 2 →  │ pos=2  │ used │ seq_id=0   │
              │  ...   │      │            │
              └────────┴──────┴────────────┘
```

**体现的关键设计：**
1. Cell 管理（`v_cells`）是全局的，跨所有 layer
2. 数据存储（`layers[il].k/v`）是分层的，每层独立
3. 通过 `il` 参数选择操作哪一层的 tensor
4. Cell 索引在所有层中保持一致，但存储的数据不同
