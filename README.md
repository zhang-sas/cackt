
FCKT Three-Innovations: Joint Aspect Extraction & Sentiment Classification

A BERT-based ABSA (Aspect-Based Sentiment Analysis) joint model that implements three innovative improvements on the original FCKT framework, enhancing aspect recall and sentiment classification accuracy.

Three Innovations

Confidence-Aware Soft Candidate Transfer

Replaces FCKT's original hard-threshold heuristic extraction with confidence-aware soft candidate transfer to improve aspect recall and reduce the erroneous deletion of candidates.

Fixed the candidate sorting key in span_annotate_candidates
No longer relies on the fixed logit_threshold=7.5; added a dynamic threshold (--candidate_threshold_mode dynamic)
Retains top-K candidate spans, each accompanied by a confidence score
Sentiment classification loss is weighted by candidate confidence
Gold span confidence is fixed at 1.0, while pseudo spans are calibrated according to the dynamic threshold

Sentiment-Conditioned Boundary Extraction

Allows sentiment to inversely constrain Aspect Extraction (AE), rather than having AE solely serve Sentiment Prediction (SP).

Defines three sentiment prototypes: neutral, positive, and negative
Calculates sentiment-aware start/end scores for each token
Generates sentiment-specific span candidates
Outputs (aspect, sentiment) jointly, rather than extracting aspect first and then classifying sentiment
Controlled via --use_sentiment_conditioned_boundary and --sentiment_prior_weight

Boundary-Aware Hard Negative Contrastive Learning

Replaces the original random negative token contrastive learning, targeting boundary confusion and same-aspect-different-sentiment issues.

Positive samples: start-end / span-sentiment pairs within gold spans
Hard negative samples: overlapping incorrect spans, other aspects in the same sentence, or aspects with the same text but different sentiments
Uses deterministic Boundary-aware Hard Negative InfoNCE, eliminating random sampling
Fixed the dimension squeezing issue with squeeze when batch_size=1 (changed to squeeze(-1))
Weight controlled via --weight_span

Project Structure

fckt_three_innovations/
├── absa/                           # Core modules for ABSA tasks
│   ├── init.py
│   ├── utils.py                    # Data loading / candidate span generation / confidence annotation ★
│   ├── run_base.py                 # Basic training utilities (optimizer / parameter loading)
│   ├── run_cls_span.py             # Evaluation functions (eval_absa, eval_ac)
│   ├── run_joint.py                # Main entry point for joint training ★
│   └── run_extract_span.py         # Independent entry point for Aspect Extraction
│
├── bert/                           # Custom BERT model modules
│   ├── init.py
│   ├── modeling.py                 # BertModel / BertConfig (based on Google BERT)
│   ├── sentiment_modeling.py       # Sentiment-conditioned boundary + contrastive loss + prototypes ★
│   ├── optimization.py             # BERT optimizer (AdamW + warmup)
│   ├── dynamic_rnn.py              # Dynamic LSTM layer
│   └── tokenization.py             # BERT tokenizer
│
├── squad/                          # SQuAD evaluation tools (reused for span extraction)
│   ├── init.py
│   ├── squad_utils.py              # Span post-processing (best_indexes, final_text)
│   └── squad_evaluate.py           # Exact match / F1 scoring
│
├── data/                           # Datasets
│   └── absa/
│       ├── laptop14_train.txt / laptop14_test.txt
│       ├── rest_total_train.txt / rest_total_test.txt
│       ├── split_laptop14_train.txt / split_rest_total_train.txt
│       ├── split_twitter1_train.txt ... split_twitter10_train.txt
│       └── length_split_*.txt
│
├── bert-base-uncased/              # BERT-base pre-trained weights (12 layers, 768 dims)
│   ├── bert_config.json / config.json
│   ├── pytorch_model.bin
│   ├── model.safetensors
│   ├── vocab.txt
│   └── tokenizer.json / tokenizer_config.json
│
├── bert-large-uncased/             # BERT-large pre-trained weights (24 layers, 1024 dims)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab.txt
│   └── tokenizer.json / tokenizer_config.json
│
├── run.py                          # Grid runner experiment entry (supports multiple parameter combinations)
├── run_twitter_sensitivity.py      # Twitter sensitivity analysis
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file

★ Files marked with a star are the core modified files for the three innovations.

Requirements
Dependency   Version   Description
Python   ≥ 3.6   Uses f-strings and type hints

PyTorch   ≥ 1.0.0   Deep learning framework

NumPy   ≥ 1.16.0   Numerical computing

Six   ≥ 1.12.0   Python 2/3 compatibility layer

CUDA   Recommended 10.0+   GPU acceleration (optional, supports --no_cuda)

Disk Space   ~5 GB   Two BERT models + datasets

Installation

Clone or unzip the project
cd fckt_three_innovations

Install dependencies
pip install -r requirements.txt

Verify environment
python -m compileall -q .
python -m absa.run_joint --help

Quick Start

python run.py
