# 对 QST 官方仓库的补丁记录

本仓库是 `https://github.com/youarespecialtome/qst.git` 的浅 clone。
由于版本老旧 + 原作者的一些 import 错误，需要对源文件打一些补丁。所有改动都应记录在本文件。

**改动原则**：
- 只做**使代码在 transformers 4.56 + Mac CPU 环境能跑通**的最小修改
- 不改业务逻辑（side network 结构、forward 流程等一律保持原样）
- AttnRes 相关改造**不写进这里**，写到我们自己的 `src/` 下

---

## Patch 1: `modeling_opt_qst.py` L21-L23 — 修复 import

**原因**：
- L21 `from QST.QSTGenerationMixin import QSTGenerationMixin` —— `QST` 这个包前缀不存在（`QSTGenerationMixin.py` 与当前文件同目录）。
- L23 `import bitsandbytes as bnb` —— Mac 无 GPU 时 bnb 装不上；且实际上本文件内并无 `bnb.xxx` 调用（死 import）。

**改动前**：
```python
from QST.QSTGenerationMixin import QSTGenerationMixin
from QSTConfig import AdapterLinear
import bitsandbytes as bnb
```

**改动后**：
```python
from QSTGenerationMixin import QSTGenerationMixin
from QSTConfig import AdapterLinear
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
```

---

## Patch 2: `QSTConfig.py` L10 — 条件化 bnb

**原因**：同 Patch 1，死 import，Mac 上装不上。

**改动前**：
```python
import bitsandbytes as bnb
```

**改动后**：
```python
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
```

---

## Patch 3: `modeling_opt_qst.py` L2044 — 修 `score_z` 维度 bug（针对 `word_embed_proj_dim ≠ hidden_size` 的 OPT 变体）

**原因**：
`score_z` 是 `QSTOPTForSequenceClassification` 里用于融合 backbone 与 side 最终 hidden 的 sigmoid 门控。
原代码写的是 `nn.Parameter(torch.zeros(config.hidden_size))`。

问题：
- `hidden_states`（backbone 最终输出）经过 `project_out` 后维度是 **`word_embed_proj_dim`**。
- `qst_hidden_states` 经过 `self.upsample` 后维度也是 **`word_embed_proj_dim`**（L2045 `self.upsample = nn.Linear(self.qst_hidden_dim, config.word_embed_proj_dim)`）。
- `self.score = nn.Linear(config.word_embed_proj_dim, num_labels)` 也要求输入 `word_embed_proj_dim`。

对 OPT-1.3B 及以上，`hidden_size == word_embed_proj_dim`，原写法恰好正确；
但 OPT 家族里小模型（如 OPT-125M 的 1024/768、OPT-350M 的 1024/512）两者不等，`score_z * qst_hidden_states` 会形状不匹配。

本项目只用 OPT-1.3B，但保留修正以保证代码在任意 OPT 尺寸上都正确。

**改动前**：
```python
self.score_z = nn.Parameter(torch.zeros(config.hidden_size)).to(llm.score.weight.device)
```

**改动后**：
```python
self.score_z = nn.Parameter(torch.zeros(config.word_embed_proj_dim)).to(llm.score.weight.device)
```

对 OPT-1.3B 这是 no-op（两维度相等），不影响当前实验。

---

## Patch 4: `modeling_opt_qst.py` L884-914 — 修 `project_in/out` 在 4-bit 量化下的形状错误

**原因**：
只有 `word_embed_proj_dim ≠ hidden_size` 的 OPT 变体才有 `project_in` / `project_out` 两个 `nn.Linear`。
对 OPT-1.3B 及以上这一条件为假，整段代码分支根本不进入（Patch 4 对 OPT-1.3B 是**死代码保护**，不改变行为）。

原 QST 代码：
```python
self.project_in = nn.Linear(..., bias=False)
self.project_in.weight = llm.project_in.weight   # 👈 问题在这
```

在 4-bit 量化加载下，`llm.project_in` 已经被 bitsandbytes 替换为 `bnb.nn.Linear4bit`，它的 `.weight` 是 `bnb.nn.Params4bit`（packed 1-D uint8 存储，不是 `[out, in]` 的 2-D 浮点矩阵）。
把这份 1-D packed 权重塞进普通 `nn.Linear` 后，forward 调朴素 `F.linear`，拿 1-D 包当 2-D 矩阵乘 → 形状错误：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (152x512 and 1x262144)
```

OPT-1.3B+ 不进入这个 if 分支（死代码），所以当前实验不受影响。补丁保留的价值在于：让代码在任意 OPT 变体上都正确。

**改动**：直接复用 `llm.project_in` / `llm.project_out` 模块本身（保留 bnb 的 4bit forward dispatch），不再拆成 `nn.Linear + copy weight`。对 fp16/fp32 加载路径等价于直接 alias，不变。

**改动前**（L884-914）：
```python
if config.word_embed_proj_dim != config.hidden_size:
    project_out_device = get_device(...)
    if isinstance(llm.project_out, nn.Linear):
        self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        self.project_out.weight = llm.project_out.weight
        self.project_out.weight.requires_grad = False
        ...
else:
    self.project_out = None

# (同样的模式用在 project_in 上)
```

**改动后**：
```python
if config.word_embed_proj_dim != config.hidden_size:
    project_out_device = get_device(...)
    self.project_out = llm.project_out    # 直接用，保留 bnb.Linear4bit
    for p in self.project_out.parameters():
        p.requires_grad = False
    self.project_out = self.project_out.to(project_out_device)
    ...
else:
    self.project_out = None

# (同样的模式用在 project_in 上)
```

---

## 非代码层的兼容性补救（在 `src/train_glue.py` 里做，不改 QST 源码）

### A. 关闭 KV cache

新 transformers 4.56 的 `OPTDecoderLayer.forward` 返回元组不再包含 cache（从 `(hidden, attn, cache)` 变为 `(hidden,)` 或 `(hidden, attn)`），
原 QST 代码 `modeling_opt_qst.py:1214` 在 `if use_cache:` 分支里索引 `layer_outputs[1]` 会 IndexError。

训练本来就不需要 KV cache，在训练脚本里加：
```python
model.config.use_cache = False
model.model.config.use_cache = False
model.model.decoder.config.use_cache = False
```
三层都设是因为子模块可能持有 config 引用；三层全关最稳妥。

### B. FutureWarning 列表（目前不 block 训练，transformers 4.58 时需要再修）

1. `past_key_value` → `past_key_values`：QST 源码在 `modeling_opt_qst.py:1189, 1203` 传的是旧名字，当前版本仍有 alias。
2. Trainer 的 `tokenizer=` → `processing_class=`：`src/train_glue.py` 已采用新名字。
3. `torch_dtype` → `dtype`：`from_pretrained` 里已采用新名字。

到 transformers 4.58 时需要：扫一遍 `modeling_opt_qst.py` 里所有 `past_key_value=` 的调用站点，统一改名。

### C. 强制 attn_implementation="eager"（绕 cuDNN SDPA 崩）

`transformers 4.56` 下 OPT 默认走 SDPA attention。某些 GPU / cuDNN 9.x / 驱动组合下，cuDNN frontend 会报：

```
RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error: No execution plans support the graph.
```

修法：`src/train_glue.py` 里加载 `llm` 时传 `attn_implementation="eager"`。此时 `OPTDecoderLayer.self_attn` 走原始 eager 实现（PyTorch 朴素 matmul + softmax），不触发 SDPA 图编译。代价是轻微速度损失，数值等价。

---

## 待观察的潜在兼容性问题（未触发前不动）

以下问题可能在真正跑训练时才暴露出来，若发生再增加 patch 条目：

1. `TrainingArguments.evaluation_strategy` 在 transformers 4.46+ 已改名 `eval_strategy`（alias 可能已被删）—— 我们直接用新名字，已 OK。
2. `AutoModelForSequenceClassification.from_pretrained(load_in_4bit=True)` 在新版本需要显式 `quantization_config=BitsAndBytesConfig(...)`（这个原代码已经做了，应该没问题；GPU 部署时再验证）。
3. `QSTConfig` 继承 `PretrainedConfig` 但用 `@dataclass` 装饰 —— 在 4.56 下行为可能微妙；如果构造时报错，改为纯 `@dataclass`。
