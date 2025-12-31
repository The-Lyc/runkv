# KV Cache：Token 元数据管理 vs Layer 数据存储

## 问题：为什么元数据按 Token 管理，但存储按 Layer 分离？

### 简短回答
**Cell 管理**和**数据存储**是两个不同的维度：
- **Cell 元数据**：记录"哪些 token 位置被占用"（全局统一）
- **实际数据**：每层对同一个 token 的处理结果不同（按层分离）

---

## 详细解释

### 1. 一个 Token 在不同 Layer 有不同的表示

在 Transformer 模型中，同一个 token 经过不同层会产生不同的隐藏状态：

```python
# 伪代码示例
token = "Hello"

# Layer 0 处理
hidden_0 = layer_0(token_embedding)
K_0, V_0 = compute_kv(hidden_0)  # Layer 0 的 K/V

# Layer 1 处理
hidden_1 = layer_1(hidden_0)
K_1, V_1 = compute_kv(hidden_1)  # Layer 1 的 K/V (不同于 Layer 0!)

# Layer 2 处理
hidden_2 = layer_2(hidden_1)
K_2, V_2 = compute_kv(hidden_2)  # Layer 2 的 K/V (又不同了!)
```

**关键：同一个 token "Hello" 在每层都有不同的 K/V 值**

### 2. Cell 索引的作用：跨层的"地址"

```cpp
// 假设我们要存储 token "Hello" (position = 5)
// v_cells 分配它到 Cell 3

v_cells[stream].pos[3] = 5;      // Cell 3 存储 position=5 的 token
v_cells[stream].seq[3].set(0);   // 属于 sequence 0

// 现在在每层都使用 Cell 3，但存储不同的数据：
layers[0].k[..., 3] = K_layer0_token5;  // Layer 0 对 token "Hello" 的 K
layers[1].k[..., 3] = K_layer1_token5;  // Layer 1 对 token "Hello" 的 K
layers[2].k[..., 3] = K_layer2_token5;  // Layer 2 对 token "Hello" 的 K
```

### 3. 具体例子：处理 3 个 Token

假设输入："How are you"

```
Token 序列:
  Token 0: "How"  (position = 0)
  Token 1: "are"  (position = 1)
  Token 2: "you"  (position = 2)

步骤 1: Cell 分配（元数据管理 - 按 Token）
  v_cells.pos[0] = 0   // Cell 0 存储 position=0
  v_cells.pos[1] = 1   // Cell 1 存储 position=1
  v_cells.pos[2] = 2   // Cell 2 存储 position=2
  
  k_idxs = [0, 1, 2]   // 3 个 token 对应的 cell 索引

步骤 2: Layer 0 前向传播
  // 计算 Layer 0 的 K/V
  k_cur_l0 = WK_0 @ hidden_0  // shape: [128, 32, 3]
  
  // 写入 Layer 0 的 cache
  cpy_k(k_cur_l0, k_idxs, il=0)
  
  执行结果：
    layers[0].k[..., 0] = k_cur_l0[..., 0]  // "How" 在 Layer 0 的 K
    layers[0].k[..., 1] = k_cur_l0[..., 1]  // "are" 在 Layer 0 的 K
    layers[0].k[..., 2] = k_cur_l0[..., 2]  // "you" 在 Layer 0 的 K

步骤 3: Layer 1 前向传播
  // 计算 Layer 1 的 K/V（基于 Layer 0 的输出）
  k_cur_l1 = WK_1 @ hidden_1  // shape: [128, 32, 3]
  
  // 写入 Layer 1 的 cache（使用相同的 cell 索引！）
  cpy_k(k_cur_l1, k_idxs, il=1)
  
  执行结果：
    layers[1].k[..., 0] = k_cur_l1[..., 0]  // "How" 在 Layer 1 的 K
    layers[1].k[..., 1] = k_cur_l1[..., 1]  // "are" 在 Layer 1 的 K
    layers[1].k[..., 2] = k_cur_l1[..., 2]  // "you" 在 Layer 1 的 K

步骤 4: Layer 2 前向传播
  // 同上...
  layers[2].k[..., 0] = k_cur_l2[..., 0]  // "How" 在 Layer 2 的 K
  layers[2].k[..., 1] = k_cur_l2[..., 1]  // "are" 在 Layer 2 的 K
  layers[2].k[..., 2] = k_cur_l2[..., 2]  // "you" 在 Layer 2 的 K
```

### 4. 内存布局可视化

```
v_cells (元数据 - 全局统一):
┌──────┬────────┬─────────┬─────────┐
│ Cell │ pos    │ seq_id  │ used    │
├──────┼────────┼─────────┼─────────┤
│  0   │   0    │  {0}    │  true   │  ← "How" 的地址
│  1   │   1    │  {0}    │  true   │  ← "are" 的地址
│  2   │   2    │  {0}    │  true   │  ← "you" 的地址
│  3   │  -1    │  {}     │  false  │  ← 未使用
│  4   │  -1    │  {}     │  false  │  ← 未使用
└──────┴────────┴─────────┴─────────┘

layers[0].k (Layer 0 数据):
┌──────┬─────────────────────────────┐
│ Cell │ 存储内容                     │
├──────┼─────────────────────────────┤
│  0   │ [Layer 0 处理 "How" 的 K]   │
│  1   │ [Layer 0 处理 "are" 的 K]   │
│  2   │ [Layer 0 处理 "you" 的 K]   │
│  3   │ [未使用的空间]               │
│  4   │ [未使用的空间]               │
└──────┴─────────────────────────────┘

layers[1].k (Layer 1 数据 - 独立的内存空间):
┌──────┬─────────────────────────────┐
│ Cell │ 存储内容                     │
├──────┼─────────────────────────────┤
│  0   │ [Layer 1 处理 "How" 的 K]   │  ← 注意：Cell 0 但内容不同！
│  1   │ [Layer 1 处理 "are" 的 K]   │
│  2   │ [Layer 1 处理 "you" 的 K]   │
│  3   │ [未使用的空间]               │
│  4   │ [未使用的空间]               │
└──────┴─────────────────────────────┘

layers[2].k (Layer 2 数据 - 又一个独立空间):
┌──────┬─────────────────────────────┐
│ Cell │ 存储内容                     │
├──────┼─────────────────────────────┤
│  0   │ [Layer 2 处理 "How" 的 K]   │  ← 同样是 Cell 0，但又不同！
│  1   │ [Layer 2 处理 "are" 的 K]   │
│  2   │ [Layer 2 处理 "you" 的 K]   │
│  3   │ [未使用的空间]               │
│  4   │ [未使用的空间]               │
└──────┴─────────────────────────────┘
```

### 5. 为什么这样设计？

#### 方案 A：如果真的完全按 Token 存储
```cpp
// 假设的"按 token 存储"方案
struct token_kv_data {
    vector<tensor> k_all_layers;  // 所有层的 K
    vector<tensor> v_all_layers;  // 所有层的 V
};

vector<token_kv_data> cache;  // 每个 cell 存储一个 token 的所有层数据

// 问题：
// 1. 读取 Layer 3 的所有 K 时，需要遍历所有 token
//    for (int i = 0; i < n_tokens; i++) {
//        k_layer3[i] = cache[i].k_all_layers[3];  // 内存不连续！
//    }
// 2. 无法利用 ggml 的高效矩阵运算
```

#### 方案 B：当前的"按 Layer 存储"方案（实际使用）
```cpp
// 每层独立的大 tensor
layers[il].k = [n_embd, kv_size, n_stream]  // 连续内存

// 优势：
// 1. 读取 Layer 3 的所有 K：直接返回 layers[3].k
// 2. 内存连续，GPU 访问高效
// 3. 可以直接用于 ggml_mul_mat 等操作
```

### 6. 代码中的体现

```cpp
// src/llama-graph.cpp:1669-1677
// 在每一层的处理中：

for (int il = 0; il < n_layers; il++) {
    // 1. 计算当前层的 K/V
    ggml_tensor * k_cur = compute_k(il);  // [head_dim, n_heads, n_tokens]
    
    // 2. 写入当前层的 cache（使用全局统一的 cell 索引）
    cpy_k(k_cur, k_idxs, il);
    //                    ↑ 指定写入哪一层
    
    // 3. 读取当前层的 cache
    ggml_tensor * k = get_k(il);
    //                      ↑ 指定读取哪一层
    
    // 4. 进行当前层的 attention 计算
    attn = compute_attention(q, k, v);
}
```

### 7. 类比：数据库表设计

这就像数据库设计中的两种方案：

**方案 A：宽表（所有数据在一起）**
```sql
CREATE TABLE kv_cache (
    token_id INT,
    layer0_k BLOB,
    layer1_k BLOB,
    layer2_k BLOB,
    ...
);
-- 查询 Layer 3 的所有数据需要扫描全表
```

**方案 B：分表（按层分离）** ← 当前实现
```sql
CREATE TABLE kv_cache_layer0 (
    cell_id INT,
    k_value BLOB
);
CREATE TABLE kv_cache_layer1 (
    cell_id INT,
    k_value BLOB
);
-- 查询 Layer 1 的所有数据：直接查一个表
```

---

## 总结

### Cell 元数据（按 Token）
- **作用**：记录哪些 cell 被占用，对应哪些 token positions
- **范围**：全局统一，所有 layer 共享
- **类比**：房间号管理系统

### 实际数据（按 Layer）
- **作用**：存储每层对每个 token 的计算结果
- **范围**：每层独立的内存空间
- **类比**：每层楼的实际房间内容

### 关键理解
```
同一个 Cell 索引（如 Cell 5）：
  - 在 v_cells 中表示：position=10 的 token
  - 在 layers[0].k[..., 5] 中存储：Layer 0 对该 token 的 K 值
  - 在 layers[1].k[..., 5] 中存储：Layer 1 对该 token 的 K 值
  - 在 layers[2].k[..., 5] 中存储：Layer 2 对该 token 的 K 值

Cell 索引是统一的"地址"，但每层存储的"内容"不同！
```
