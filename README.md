# FCKT Three-Innovations: Joint Aspect Extraction & Sentiment Classification

基于 BERT 的 ABSA（Aspect-Based Sentiment Analysis）联合模型，在原始 FCKT 框架上实现了三个创新改进，提升 aspect 召回率和情感分类准确度。

## 三个创新点

### 1. 置信度感知软候选迁移 (Confidence-Aware Soft Candidate Transfer)

替代 FCKT 原始的硬阈值 heuristic extraction，通过置信度感知的软候选迁移提高 aspect recall，减少候选被误删。

- 修复 `span_annotate_candidates` 中的候选排序 key
- 不再依赖固定 `logit_threshold=7.5`，新增动态阈值（`--candidate_threshold_mode dynamic`）
- 保留 top-K 候选 span，每个 span 附有 confidence 分数
- 情感分类 loss 按 candidate confidence 加权
- gold span 的 confidence 固定为 1.0，pseudo span 按动态阈值校准

### 2. 情感条件化边界抽取 (Sentiment-Conditioned Boundary Extraction)

让 sentiment 反向约束 Aspect Extraction，而不是只让 AE 服务 SP。

- 定义三个 sentiment prototypes：**neutral**、**positive**、**negative**
- 对每个 token 计算 sentiment-aware start/end score
- 生成 sentiment-specific span candidates
- 最终输出 `(aspect, sentiment)`，而非先 aspect 后 sentiment
- 支持 `--use_sentiment_conditioned_boundary` 和 `--sentiment_prior_weight` 控制

### 3. 边界感知难负样本对比学习 (Boundary-Aware Hard Negative Contrastive Learning)

替代原随机负样本 token contrastive learning，让对比学习针对边界混淆和同类异情感问题。

- **正样本**：gold span 内 start-end / span-sentiment
- **难负样本**：重叠错误 span、同句其他 aspect、同类异情感 aspect
- 使用确定性 Boundary-aware Hard Negative InfoNCE，不再随机抽样
- 修复 `squeeze` 在 batch_size=1 时的维度挤压问题（改为 `squeeze(-1)`）
- 通过 `--weight_span` 控制权重

## 项目结构

```
fckt_three_innovations/
├── absa/                           # ABSA 任务核心模块
│   ├── __init__.py
│   ├── utils.py                    # 数据读取/候选span生成/置信度标注 ★
│   ├── run_base.py                 # 基础训练工具（optimizer/参数加载）
│   ├── run_cls_span.py             # 评估函数（eval_absa, eval_ac）
│   ├── run_joint.py                # 联合训练主入口 ★
│   └── run_extract_span.py         # Aspect extraction 独立入口
│
├── bert/                           # 自定义 BERT 模型模块
│   ├── __init__.py
│   ├── modeling.py                 # BertModel / BertConfig（基于 Google BERT）
│   ├── sentiment_modeling.py       # 情感条件化边界 + 对比loss + prototypes ★
│   ├── optimization.py             # BERT 优化器（AdamW + warmup）
│   ├── dynamic_rnn.py              # 动态 LSTM 层
│   └── tokenization.py             # BERT tokenizer
│
├── squad/                          # SQuAD 评估工具（复用于 span 抽取）
│   ├── __init__.py
│   ├── squad_utils.py              # Span 后处理（best_indexes, final_text）
│   └── squad_evaluate.py           # Exact match / F1 评分
│
├── data/                           # 数据集
│   └── absa/
│       ├── laptop14_train.txt / laptop14_test.txt
│       ├── rest_total_train.txt / rest_total_test.txt
│       ├── split_laptop14_train.txt / split_rest_total_train.txt
│       ├── split_twitter1_train.txt ... split_twitter10_train.txt
│       └── length_split_*.txt
│
├── bert-base-uncased/              # BERT-base 预训练权重（12层, 768维）
│   ├── bert_config.json / config.json
│   ├── pytorch_model.bin
│   ├── model.safetensors
│   ├── vocab.txt
│   └── tokenizer.json / tokenizer_config.json
│
├── bert-large-uncased/             # BERT-large 预训练权重（24层, 1024维）
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab.txt
│   └── tokenizer.json / tokenizer_config.json
│
├── run.py                          # Grid runner 实验入口（支持多参数组合）
├── run_tweets_10fold.py            # Twitter 10折交叉验证（防泄漏拆分）
├── run_restaurant_4ablation_seed58.py  # Restaurant 消融实验
├── run_twitter_sensitivity.py      # Twitter 敏感度分析
│
├── README_THREE_INNOVATIONS.md     # 原始英文快速参考
├── 已实现的三个创新点.md             # 详细中文技术说明
├── requirements.txt                # Python 依赖
└── README.md                       # 本文件
```

> ★ 标记的文件是本次三个创新点的核心修改文件。

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.6 | 使用 f-string 和 type hints |
| PyTorch | ≥ 1.0.0 | 深度学习框架 |
| NumPy | ≥ 1.16.0 | 数值计算 |
| Six | ≥ 1.12.0 | Python 2/3 兼容层 |
| CUDA | 推荐 10.0+ | GPU 加速（可选，支持 `--no_cuda`） |
| 磁盘空间 | ~5 GB | 两个 BERT 模型 + 数据集 |

### 安装

```bash
# 1. 克隆或解压项目
cd fckt_three_innovations

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证环境
python -m compileall -q .
python -m absa.run_joint --help
```

### BERT 权重说明

项目已包含 `bert-base-uncased` 和 `bert-large-uncased` 两个预训练模型的完整文件：
- `vocab.txt` — 词表
- `pytorch_model.bin` — PyTorch 权重
- `config.json` / `bert_config.json` — 模型配置

> **注意**：如果使用 `bert-large-uncased`，命令行参数 `--bert_config_file` 应指向 `bert-large-uncased/config.json`（该目录没有单独的 `bert_config.json`）。

## 快速开始

### 单次训练与评估

```bash
python -m absa.run_joint \
  --data_dir data/absa \
  --train_file split_laptop14_train.txt \
  --predict_file laptop14_test.txt \
  --bert_config_file bert-large-uncased/config.json \
  --vocab_file bert-large-uncased/vocab.txt \
  --init_checkpoint bert-large-uncased/pytorch_model.bin \
  --output_dir out/laptop14_baseline \
  --num_train_epochs 100 \
  --train_batch_size 16 \
  --predict_batch_size 16 \
  --learning_rate 2e-5 \
  --max_seq_length 85 \
  --random_train 0.9 \
  --candidate_threshold_mode dynamic \
  --weight_span 1e-7 \
  --do_train --do_predict
```

### 推荐首轮运行（完整参数）

```bash
python -m absa.run_joint \
  --do_train --do_predict \
  --candidate_threshold_mode dynamic \
  --random_train 0.9 \
  --expectation_start_step 0 \
  --n_best_size 10 \
  --weight_span 1e-7
```

## Grid Runner 批量实验

```bash
# 多参数网格搜索
python run.py \
  --data_dir data/absa \
  --train_file split_laptop14_train.txt \
  --predict_file laptop14_test.txt \
  --candidate_threshold_mode dynamic \
  --random_train 0.9 \
  --learning_rate 2e-5 \
  --weight_span 1e-7 \
  --epochs 100 \
  --seed 58 \
  --tag my_experiment

# 阈值扫描
python run.py --tag th_sweep \
  --logit_threshold 7.0,7.5,8.0 \
  --random_train 0.9 --weight_span 1e-7

# weight_span 扫描
python run.py --tag ws_sweep \
  --logit_threshold 7.5 \
  --random_train 0.9 --weight_span 0,1e-8,1e-7,1e-6
```

## 10 折交叉验证（Twitter 数据集）

```bash
python run_tweets_10fold.py \
  --data_dir data/absa \
  --bert_config_file bert-large-uncased/config.json \
  --vocab_file bert-large-uncased/vocab.txt \
  --init_checkpoint bert-large-uncased/pytorch_model.bin \
  --output_root out/twitter_10fold \
  --epochs 30 \
  --candidate_threshold_mode dynamic \
  --use_ac_confidence_filter True \
  --folds 1-10
```

脚本会自动：
1. 合并重复句子记录（防止跨子集泄露）
2. 按 sentiment 分布做分层 train/dev 拆分
3. 10 折训练 → 汇总 mean ± std 到 `cv_summary.json` 和 `cv_report.txt`

## 消融实验

五个关键消融实验，分别验证各创新点的贡献：

| # | 实验 | 命令 |
|---|------|------|
| 1 | **固定阈值基线** | `--candidate_threshold_mode fixed --logit_threshold 7.5` |
| 2 | **移除情感条件化边界** | `--use_sentiment_conditioned_boundary false` |
| 3 | **直接情感特定输出** | `--direct_sentiment_boundary_output true` |
| 4 | **移除软伪候选迁移** | `--disable_expectation` |
| 5 | **移除 Hard Negative 对比** | `--weight_span 0` |

## 核心参数说明

### 候选生成与置信度（创新点 1）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--candidate_threshold_mode` | str | `dynamic` | 候选过滤模式：`fixed` / `dynamic` / `none` |
| `--logit_threshold` | float | `7.5` | fixed 模式下的硬阈值 |
| `--dynamic_threshold_scale` | float | `0.5` | dynamic 模式的缩放系数 |
| `--candidate_min_keep` | int | `1` | 最少保留候选数 |
| `--min_candidate_confidence` | float | `0.55` | 候选最小置信度 |
| `--confidence_temperature` | float | `1.0` | 置信度温度系数 |
| `--n_best_size` | int | `8` | 每个位置保留的 top-N span |

### 情感条件化（创新点 2）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_sentiment_conditioned_boundary` | bool | `False` | 启用情感条件化边界 |
| `--sentiment_prior_weight` | float | `0.0` | 候选情感先验融合权重 |
| `--direct_sentiment_boundary_output` | bool | `False` | 直接输出情感特定边界 |

### 训练策略

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--random_train` | float | `0.9` | gold branch 概率（非 train/test 比例） |
| `--expectation_start_step` | int | `200` | 延迟 expectation 分支启用步数 |
| `--disable_expectation` | flag | — | 完全关闭 expectation 分支 |
| `--train_pseudo_other` | bool | `False` | pseudo 分支是否包含 other 标签 |

### AC 后处理

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_ac_confidence_filter` | bool | `True` | 启用 AC 置信度过滤 |
| `--ac_min_confidence` | float | `0.45` | AC 分类最低置信度 |
| `--ac_other_margin` | float | `0.03` | other 类别的 margin |
| `--drop_other_predictions` | bool | `True` | 评估时丢弃 other 预测 |

### 一致性解码

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_consistency_decoding` | bool | `True` | 启用一致性解码 |
| `--consistency_pool_multiplier` | int | `2` | 候选池倍增系数 |
| `--consistency_final_top_k` | int | `5` | 最终保留 top-K |
| `--use_pair_verifier` | bool | `True` | 启用 pair verifier |
| `--pair_verifier_alpha` | float | `1.0` | verifier 的分数融合权重 |

## 数据格式

数据集采用 ABSA 标准格式，每行为一个样本：

```
<原始句子>####<word1>=<tag1> <word2>=<tag2> ...
```

其中 tag 取值：
- `O` — 非 aspect 词
- `T-POS` — 正面情感 aspect 词
- `T-NEG` — 负面情感 aspect 词
- `T-NEU` — 中性情感 aspect 词

示例：
```
The battery life is amazing####The=O battery=T-POS life=T-POS is=O amazing=O
```

## 支持的模型

| 模型 | hidden_size | layers | heads | 参数量 |
|------|-------------|--------|-------|--------|
| `bert-base-uncased` | 768 | 12 | 12 | ~110M |
| `bert-large-uncased` | 1024 | 24 | 16 | ~340M |

## 评估指标

- **P_ae / R_ae / F1_ae** — Aspect Extraction 的精确率/召回率/F1
- **Acc_ac** — Aspect Sentiment Classification 准确率
- **P_all / R_all / F1_all** — Joint task（正确 aspect + 正确 sentiment）的综合指标

## 常见问题

**Q: 如何切换使用 bert-base 模型？**
```bash
--bert_config_file bert-base-uncased/bert_config.json \
--vocab_file bert-base-uncased/vocab.txt \
--init_checkpoint bert-base-uncased/pytorch_model.bin
```

**Q: 没有 GPU 可以跑吗？**
可以，加上 `--no_cuda` 参数，但 BERT-large 在 CPU 上训练会非常慢。

**Q: `bert_config.json` 找不到？**
bert-large-uncased 目录下配置文件名是 `config.json`，使用时请用 `--bert_config_file bert-large-uncased/config.json`。

**Q: 如何只做推理不做训练？**
去掉 `--do_train`，只保留 `--do_predict`。

## 许可证

本项目基于 Google BERT 代码和 FCKT 模型修改。原始 BERT 代码版权归 Google AI Language Team 和 HuggingFace Inc. 所有，采用 Apache License 2.0。

## 引用

本项目的三个创新点描述详见：
- `已实现的三个创新点.md` — 中文完整技术说明
- `README_THREE_INNOVATIONS.md` — 英文快速参考
