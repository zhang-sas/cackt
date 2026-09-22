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
├── run_twitter_sensitivity.py      # Twitter 敏感度分析
│
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


## 快速开始


```bash
python run.py


