# Automatic Clickbait Detection

A comparative study of machine learning and deep learning approaches to detecting clickbait headlines, paired with a user-facing A/B test measuring whether showing a model's clickbait score actually changes what people choose to read.

This was originally a postgraduate dissertation project (MSc Data Science, University of Southampton). The code here has been cleaned up and restructured for general reference — preprocessing notebooks have been converted into standalone scripts and reorganised around the experiments rather than the order they were run in.

## What's here

**Detection models** (`src/`) — classical ML, a CNN, and fine-tuned RoBERTa, trained and evaluated on the [Webis Clickbait Corpus 2017](https://webis.de/data/webis-clickbait-17.html).

**A/B test** (`ab_test/`) — a small web app that shows participants headlines either with or without a clickbait-likelihood badge, and records which articles they choose to read.

## Results summary

| Approach | Test Accuracy | Test F1 (macro) |
|---|---|---|
| Logistic Regression (handcrafted features) | 0.74 | 0.62 |
| Logistic Regression (TF-IDF) | 0.76 | 0.72 |
| XGBoost (TF-IDF, tuned) | 0.76 | 0.75 |
| CNN (raw text) | 0.80 | 0.76 |
| RoBERTa (title only) | 0.73 | 0.69 |
| **RoBERTa + count features (best)** | **0.75** | **0.71** |

Headline length turned out to be a more useful predictor than punctuation or slang. Adding handcrafted style features into RoBERTa generally hurt test performance (overfitting to validation), with the exception of basic count features. Full breakdown, confusion matrices, and the A/B test results are in `results/` and were originally written up in the dissertation report.

## Project structure

```
acbd/
├── data/                       # not included — see Data section below
├── src/
│   ├── preprocessing.py        # load corpus, extract handcrafted features
│   ├── models_classical.py     # LR, SVM, RF, DT, NB, XGBoost (+ TF-IDF, tuning, balancing)
│   ├── models_deep.py          # CNN on raw title text
│   ├── models_roberta.py       # fine-tuned RoBERTa, 4 feature-fusion variants
│   └── evaluate.py             # aggregates results, builds comparison chart
├── results/                    # output CSVs and charts (generated)
├── ab_test/                    # A/B testing web interface
└── requirements.txt
```

## Data

This project uses the **Webis Clickbait Corpus 2017**, available from [webis.de](https://webis.de/data/webis-clickbait-17.html) (registration required). It is not redistributed here.

To run the pipeline, place the corpus under `data/` as:

```
data/
├── train/
│   ├── instances.jsonl
│   └── truth.jsonl
├── test/
│   ├── instances.jsonl
│   └── truth.jsonl
└── all_slangs.csv
```

`all_slangs.csv` is a custom slang/abbreviation dictionary (LOL, JK, WTH, etc.) used for one of the handcrafted features — not part of the original corpus.

## Running it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 1. Extract handcrafted features (also creates the train/val split)
python src/preprocessing.py

# 2. Classical ML — handcrafted features, TF-IDF, tuning, balanced datasets
python src/models_classical.py

# 3. CNN (requires TensorFlow)
python src/models_deep.py

# 4. RoBERTa (requires PyTorch + transformers, GPU strongly recommended)
python src/models_roberta.py

# 5. Aggregate everything into one comparison table/chart
python src/evaluate.py
```

`models_roberta.py` fine-tunes a full transformer four times (once per feature variant) and will be very slow on CPU. A single GPU (even a free Colab T4) brings this down to a reasonable runtime.

## Method notes

- **Class imbalance**: the corpus is roughly 76% non-clickbait. Classical models use `class_weight="balanced"`; a separate experiment retrains on five undersampled, balanced training sets (different random seeds) to check robustness.
- **Feature ablation**: handcrafted features are also evaluated in isolation (count-based, question-based, style-based) to see which signals actually carry predictive value — count-based features (headline length) consistently outperformed punctuation/slang-based ones.
- **Splits**: train/val split uses `random_state=47`, stratified on the label, throughout — this needs to stay consistent across scripts since `models_roberta.py` re-derives the same split to align raw text with pre-computed features.

## A/B test

See [`ab_test/README.md`](ab_test/README.md) for details on the user study, including an honest note on how participant data was actually collected.

## Acknowledgements

Built with scikit-learn, XGBoost, TensorFlow/Keras, and Hugging Face Transformers (`roberta-base`). Dataset: Webis Clickbait Corpus 2017 (Potthast et al.).
