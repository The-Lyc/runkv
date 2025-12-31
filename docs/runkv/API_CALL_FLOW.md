# llama.cpp 完整推理请求的 API 调用流程

本文档说明在 llama.cpp 中提交并执行一个完整推理请求需要调用的核心函数。

## 一、初始化阶段（一次性）

### 1. 加载后端
```cpp
ggml_backend_load_all();
```
- **作用**: 加载所有可用的计算后端（CPU, CUDA, Metal等）
- **调用时机**: 程序启动时调用一次

### 2. 加载模型
```cpp
// 配置模型参数
llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 32;  // GPU层数

// 加载模型
llama_model * model = llama_model_load_from_file(model_path, model_params);
```
- **作用**: 从 GGUF 文件加载模型权重
- **关键参数**: 
  - `n_gpu_layers`: 卸载到GPU的层数
  - `vocab_only`: 是否只加载词表

### 3. 创建推理上下文
```cpp
// 配置上下文参数
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048;      // KV cache 大小（总 cells 数）
ctx_params.n_batch = 512;     // 逻辑 batch 大小
ctx_params.n_ubatch = 128;    // 物理 ubatch 大小
ctx_params.n_seq_max = 4;     // 最大并发序列数

// 创建上下文（分配 KV cache）
llama_context * ctx = llama_init_from_model(model, ctx_params);
```
- **作用**: 创建推理上下文，分配 KV cache 内存
- **关键参数**:
  - `n_ctx`: KV cache cells 总数（影响可支持的最大文本长度）
  - `n_batch`: 每次可处理的最大 token 数
  - `n_seq_max`: 最大并发序列数（影响 unified vs streaming 模式）

### 4. 创建采样器
```cpp
// 创建采样器链
auto sparams = llama_sampler_chain_default_params();
llama_sampler * smpl = llama_sampler_chain_init(sparams);

// 添加采样策略
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));  // 最终采样器
```
- **作用**: 配置 token 采样策略
- **常见采样器**:
  - `top_k`: Top-K 采样
  - `top_p`: Top-P (nucleus) 采样
  - `temp`: 温度缩放
  - `greedy`: 贪心采样（总是选最大概率）
  - `dist`: 按分布采样

## 二、单次请求处理（每个请求）

### 5. Tokenize 输入文本
```cpp
const llama_vocab * vocab = llama_model_get_vocab(model);

// 方法1: 两步法（推荐）
// 第一步：获取 token 数量
int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), 
                                NULL, 0, true, true);

// 第二步：分配空间并 tokenize
std::vector<llama_token> tokens(n_tokens);
llama_tokenize(vocab, prompt.c_str(), prompt.size(), 
               tokens.data(), n_tokens, true, true);
```
- **作用**: 将文本转换为 token ID 序列
- **参数说明**:
  - `add_special=true`: 添加 BOS (beginning of sequence) token
  - `parse_special=true`: 解析特殊 token (如 `<|endoftext|>`)

### 6. 创建 Batch
```cpp
// 方法1: 使用便捷函数（单序列）
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

// 方法2: 手动创建（支持多序列）
llama_batch batch = llama_batch_init(n_batch_max, 0, n_seq_max);
```
- **作用**: 创建 batch 容器
- **Batch 结构**:
  - `token[]`: token ID 数组
  - `pos[]`: 每个 token 的位置（position）
  - `seq_id[][]`: 每个 token 所属的序列 ID 列表
  - `logits[]`: 是否需要输出该 token 的 logits

### 7. 填充 Batch（多序列/手动控制时）
```cpp
// 使用 common 辅助函数
for (int i = 0; i < n_tokens; i++) {
    common_batch_add(batch, tokens[i], i, { seq_id }, false);
}
// 最后一个 token 需要输出 logits
batch.logits[batch.n_tokens - 1] = true;
```
- **作用**: 向 batch 添加 token
- **参数说明**:
  - `token`: token ID
  - `pos`: position（通常从 0 开始递增）
  - `seq_ids`: 该 token 所属的序列 ID 列表（支持一个 token 属于多个序列）
  - `logits`: 是否需要输出该 token 的 logits（通常只有最后一个需要）

### 8. 执行推理（Prefill/Decode）
```cpp
int ret = llama_decode(ctx, batch);
if (ret != 0) {
    // 错误处理
    // ret=1: KV cache 空间不足
    // ret=2: 被中止
    // ret<0: 致命错误
}
```
- **作用**: **核心推理函数**，执行 transformer 前向传播
- **内部流程**:
  1. 调用 `llama_kv_cache::find_slot()` 寻找可用的 KV cache cells
  2. 分 ubatch 执行计算（每个 ubatch ≤ n_ubatch tokens）
  3. 调用 `llama_kv_cache::apply_ubatch()` 更新 cell 元数据
  4. 计算 K/V 并存入 KV cache
  5. 计算 attention 和 FFN
  6. 输出 logits（对于 `logits[i]=true` 的 token）
- **返回值**:
  - `0`: 成功
  - `1`: KV cache 空间不足
  - `2`: 被中止（通过 abort callback）
  - `<0`: 致命错误

### 9. 获取 Logits
```cpp
// 获取最后一个 token 的 logits
float * logits = llama_get_logits_ith(ctx, -1);  // -1 表示最后一个

// 或者获取第 i 个输出 token 的 logits
float * logits_i = llama_get_logits_ith(ctx, i);
```
- **作用**: 获取模型输出的 logits（词表概率分布）
- **注意**: 只有 `batch.logits[i]=true` 的 token 才会有输出

### 10. 采样下一个 Token
```cpp
llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
```
- **作用**: 根据 logits 和采样策略选择下一个 token
- **参数说明**:
  - `ctx`: 上下文（包含 logits）
  - `-1`: 使用最后一个 token 的 logits

### 11. 检查结束条件
```cpp
if (llama_vocab_is_eog(vocab, new_token)) {
    // 遇到 EOS/EOT，结束生成
    break;
}
```
- **作用**: 检查是否遇到结束 token

### 12. 将 Token 转换为文本（可选）
```cpp
char buf[128];
int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
std::string text(buf, n);
printf("%s", text.c_str());
```
- **作用**: 将 token ID 转换为文本输出

### 13. 准备下一轮 Decode
```cpp
// 创建新 batch（只包含采样的新 token）
batch = llama_batch_get_one(&new_token, 1);

// 继续循环到步骤 8
```

## 三、多序列管理（并发请求）

### 14. 管理 KV Cache 序列
```cpp
// 获取 memory 对象
llama_memory_t mem = llama_get_memory(ctx);

// 删除序列（释放 KV cache）
llama_memory_seq_rm(mem, seq_id, -1, -1);

// 复制序列（用于 beam search）
llama_memory_seq_cp(mem, src_seq_id, dst_seq_id, -1, -1);

// 保留序列的某个范围
llama_memory_seq_keep(mem, seq_id);

// 查询序列的位置范围
llama_pos min_pos = llama_memory_seq_pos_min(mem, seq_id);
llama_pos max_pos = llama_memory_seq_pos_max(mem, seq_id);
```

## 四、清理阶段

### 15. 释放资源
```cpp
// 按相反顺序释放
llama_sampler_free(smpl);    // 释放采样器
llama_batch_free(batch);     // 释放 batch
llama_free(ctx);             // 释放上下文（释放 KV cache）
llama_model_free(model);     // 释放模型
```

## 完整调用流程图

```
初始化阶段:
  ggml_backend_load_all()
       ↓
  llama_model_load_from_file()
       ↓
  llama_init_from_model()  ← 分配 KV cache
       ↓
  llama_sampler_chain_init() + llama_sampler_chain_add()

请求处理循环:
  llama_tokenize() ← 文本 → tokens
       ↓
  llama_batch_get_one() / common_batch_add()
       ↓
  ┌─→ llama_decode() ← ★★★ 核心推理 ★★★
  │    ├─ find_slot()      (找 KV cache 空间)
  │    ├─ apply_ubatch()   (更新 cell 元数据)
  │    └─ 计算 transformer (K/V 写入 cache)
  │        ↓
  │   llama_get_logits_ith()
  │        ↓
  │   llama_sampler_sample()
  │        ↓
  │   llama_vocab_is_eog() ─→ 结束? → 退出
  │        ↓ 否
  │   llama_token_to_piece() → 输出文本
  │        ↓
  │   llama_batch_get_one() (准备下一个 token)
  └────┘

多序列管理（可选）:
  llama_get_memory()
       ↓
  llama_memory_seq_rm() / _cp() / _keep()

清理阶段:
  llama_sampler_free()
       ↓
  llama_batch_free()
       ↓
  llama_free()
       ↓
  llama_model_free()
```

## 关键函数总结

| 函数 | 作用 | 调用频率 | 是否核心 |
|------|------|----------|----------|
| `llama_decode()` | 执行推理 | 每个 token | ★★★ 核心 |
| `llama_get_logits_ith()` | 获取输出 | 每个 token | ★★★ 核心 |
| `llama_sampler_sample()` | 采样 token | 每个 token | ★★★ 核心 |
| `llama_tokenize()` | 文本转 tokens | 每个请求开始 | ★★ 重要 |
| `llama_batch_get_one()` | 创建 batch | 每个 token | ★★ 重要 |
| `common_batch_add()` | 填充 batch | 多序列场景 | ★ 辅助 |
| `llama_memory_seq_rm()` | 清理 KV cache | 请求结束 | ★ 管理 |
| `llama_token_to_piece()` | token 转文本 | 输出时 | - 可选 |

## 注意事项

1. **`llama_decode()` 是唯一执行推理的函数**
   - 其他函数都是准备数据或处理结果
   - KV cache 的分配、更新都在这个函数内部完成

2. **Batch 的 `logits[]` 标志很重要**
   - 只有设置为 `true` 的 token 才会输出 logits
   - Prefill 阶段通常只需要最后一个 token 的 logits
   - Decode 阶段每次只有一个 token，需要设置其 logits=true

3. **Position 必须正确**
   - Prefill: 从 0 开始递增
   - Decode: 使用上一个 token 的 position + 1

4. **多序列场景需要手动管理 seq_id**
   - 使用 `common_batch_add()` 更方便
   - 不同请求使用不同的 seq_id
   - 请求结束后调用 `llama_memory_seq_rm()` 释放空间

5. **错误处理**
   - `llama_decode()` 返回 1 表示 KV cache 不足
   - 需要清理旧序列或增大 `n_ctx`
