# Transformer 5-plex dLAMP 改动记录

## 实验背景

目标：用 Transformer 对 5-plex dLAMP 数据集做 10-fold 分层交叉验证分类。  
基线参考：KNN-ACA 模型在同数据集上 **91%** 准确率。  
数据集：`5Plex_dLAMP_oldData.csv`（54186 样本，5 类：ad/c19/ia/ib/kp，序列长度 35）  
评估方式：每次只跑 **1 fold 30k iter (cosine)** 做快速验证，确认有效再跑完整 10 fold。

---

## 实验追踪表（Fold 1 准确率）

| # | 描述 | 改动文件 | Fold 1 | 状态 |
|---|---|---|---|---|
| 0 | 原始 inv 10k | — | 89.17% | ✅ 完成 |
| 1 | inv → cosine_warmup，30k | lr_schedule.py, main_cv_5plex.py | 95.24% | ✅ 完成 |
| 2 | Fix d_ff 128→512 + PE regular→original | positionwiseFeedForward.py, main_cv_5plex.py, transformer.py | 95.81% | ✅ 完成 |
| 3 | 数据归一化 standard | main_cv.py | 95.87% | ✅ 完成 |
| 4 | 加权 CrossEntropy | train_baseline.py | ? | 🔄 进行中 |
| 5 | Mean Pooling 替代 Flatten | transformer.py | ? | ⏳ 待做 |
| 6 | 修复注意力缩放 sqrt(K)→sqrt(q) | multiHeadAttention.py | ? | ⏳ 待做 |
| 7 | 数据增强（Jitter + Magnitude warp） | data_loader.py | ? | ⏳ 待做 |
| — | 统一脚本（main_cv.py 合并三五plex） | main_cv.py | — | ✅ 完成 |

**当前基准（已实施 #1+#2+#3）：95.87%（Fold 1，cosine 30k）**

---

## 已完成的改动详情

### 改动 1：新增 5-plex 训练脚本

**文件**：`Code_David_Transformer/main_cv_5plex.py`（新建）

**原因**：原始 `main_cv.py` 硬编码了 3 类（HAdv/IAV/IBV）和序列长度 473，无法直接用于 5-plex 数据集。

**改动内容**：
- `CLASS_NUM = 5`，`ACTIVITIES = ['ad', 'c19', 'ia', 'ib', 'kp']`
- `DEFAULT_SEQ_LENGTH = 35`
- 默认数据文件指向 `5Plex_dLAMP_oldData_processed.csv`
- 输出目录前缀改为 `5plex_cv_*` 以区分不同实验

---

### 改动 2：数据预处理

**文件**：`data/5Plex_dLAMP_oldData_processed.csv`（新建）

**原因**：原始 CSV 没有 `Target_cat` 整数列，且含有 `Unnamed: 0` 等元数据列会被 `filter(regex=r'\d+\.?\d*')` 误匹配。

**改动内容**：
- 添加 `Target_cat` 列（label encoding：ad=0, c19=1, ia=2, ib=3, kp=4）
- 只保留 Target、Target_cat 和曲线列（`1.0` 到 `35.0`）

---

### 改动 3：LR 调度器从 `inv` 改为 `cosine_warmup`

**文件**：
- `src/utils/lr_schedule.py`：新增 `cosine_warmup_lr_scheduler` 函数
- `main_cv_5plex.py`：`build_config` 中切换到 `cosine_warmup`，参数 `warmup_iters=500, lr_min=1e-6`

**原因**：`inv` 调度器来自 DANN 域适应论文，专为 10000 步设计。在 5000 步时 LR 已衰减到初始值的 26%，之后无限小步爬行，永远无法真正收敛。表现为：最优 checkpoint 总在最后几步，需要 50k+ 迭代才能达到 cosine 30k 的效果。

**cosine_warmup 设计**：
- 0→500步（warmup）：LR 从 0 线性升至 lr_max
- 500→T_max步：LR 按余弦从 lr_max 降至 lr_min（T_max 处自然归零）

**效果**：cosine 30k (95.24%) > inv 50k (93.17%)

---

### 改动 4：修复 PositionwiseFeedForward 宽度 Bug

**文件**：`Code_David_Transformer/tst/positionwiseFeedForward.py:24`

```python
# 改前（Bug）：d_ff = d_model，FFN 没有扩展维度
def __init__(self, d_model: int, d_ff: Optional[int] = 128):

# 改后：标准 Transformer d_ff = 4 × d_model = 512
def __init__(self, d_model: int, d_ff: Optional[int] = 512):
```

**原因**：标准 Transformer FFN 先将 d_model 扩展到 4× 学习非线性特征再压缩回来。原来 128→128 直连，相当于一个普通线性层，严重削弱了表达能力。

---

### 改动 5：修复位置编码 Bug（regular → original PE）

**文件**：
- `main_cv_5plex.py`：`'pe': 'regular'` → `'pe': 'original'`
- `src/models/transformer.py:132-135`：只对 `pe='regular'` 时保存 `_pe_period`，避免向 `generate_original_PE` 传入多余的 `period` 参数

**Bug**：`generate_regular_PE` 把同一个标量复制 128 份作为 PE，每个维度编码完全相同，模型无法区分不同位置。

`generate_original_PE` 每维度用不同频率的正弦/余弦波，是标准实现。

---

## 待实施改动详情

### 改动 6（下一个）：数据归一化

**文件**：`main_cv_5plex.py:175`

```python
# 改前
use_normalize='None',

# 改后
use_normalize='standard',
```

**原因**：dLAMP 原始信号值域宽，未归一化导致 embedding 层 `Linear(1, 128)` 的每个输入标量量级差异大，梯度更新不稳定。StandardScaler 将每个特征归一化到均值 0、标准差 1，让模型在稳定的数值范围内学习。注意：Scaler 已在 `generate_cv_data_loader` 中正确实现，只 fit 训练集，避免数据泄露。

**预期收益**：+1~2%

---

### 改动 7：加权 CrossEntropy

**文件**：`src/training/train_baseline.py:99`

```python
# 改前
classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())

# 改后（权重 = 总样本数 / (类别数 × 该类样本数)，归一化）
# ad:12353  c19:6124  ia:14068  ib:14197  kp:7444  total:54186
weights = torch.tensor([0.874, 1.763, 0.767, 0.761, 1.454], device=device)
classifier_loss = nn.CrossEntropyLoss(weight=weights)(outputs_source, labels_source.long())
```

**原因**：当前 Fold 1 各类 F1（cosine+fix）：ib 99%、ia 95%、ad 97%，而 c19 仅 87%、kp 91%。c19 和 kp 样本少（各占 11%/14%），模型偏向多数类。加权后少数类的 loss 贡献增大，迫使模型更认真学习难分类别。

**预期收益**：c19/kp +3~5%，整体 +0.5~1%

---

### 改动 8：Mean Pooling 替代 Flatten

**文件**：`src/models/transformer.py:144-151`（classifier 部分）

```python
# 改前：Flatten 后输入 d_model×seq_length = 128×35 = 4480 维
nn.Flatten(),
nn.Linear(4480, 128),

# 改后：对时间维度做平均，直接得到 128 维表示
# 在 forward 中: x = encoding.mean(dim=1)  # (batch, 128)
# classifier 改为: nn.Linear(128, 128)
```

**原因**：4480→128 的压缩损失大量空间结构信息，且参数量大（约 57 万参数），容易过拟合。Mean pooling 只有 128×128=1.6 万参数，迫使模型在 attention 阶段提炼足够的语义信息，泛化更好。

**预期收益**：+0.5~1%

---

### 改动 9：修复注意力缩放因子

**文件**：`Code_David_Transformer/tst/multiHeadAttention.py:91`

```python
# 改前：用序列长度 K=35 做缩放
self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)

# 改后：用 query 维度 q=8 做缩放（标准做法）
self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._q)
# 需要在 __init__ 中保存: self._q = q
```

**原因**：标准 Scaled Dot-Product Attention 用 `sqrt(d_k)` 缩放，防止点积值过大导致 softmax 梯度消失。d_k = q = 8，`sqrt(8) ≈ 2.83`。原来用 `sqrt(K) = sqrt(35) ≈ 5.92`，缩放过度，attention 分布过于平坦，模型难以学到 sharp 的注意力模式。

**预期收益**：+0.2~0.5%

---

### 改动 10：数据增强

**文件**：`src/data/data_loader.py`（新增 augment 函数，在训练 loader 中使用）

**方案**：
1. **Jitter**：训练时对输入加小高斯噪声（σ=0.02）
2. **Magnitude warp**：随机缩放振幅（0.9~1.1 倍）

**原因**：35 步的短序列信息量有限，增强能让模型见到更多样的输入，特别有助于少数类（c19/kp）的泛化。增强只在训练阶段使用，测试集不做任何变换。

**预期收益**：+1~2%
