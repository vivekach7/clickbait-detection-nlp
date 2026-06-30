"""
models_roberta.py
------------------
Fine-tunes RoBERTa-base on targetTitle text, with optional fusion
of handcrafted features (count-based, style-based, or all) into
the classification head.

Variants tested (matching the dissertation):
    1. targetTitle only
    2. targetTitle + Count-based features (chars, words)
    3. targetTitle + Style-based features (exclaim, upper-ratio, slang, adj)
    4. targetTitle + all 8 handcrafted features

Requires a GPU for reasonable training time. Falls back to CPU
(slow) if none available.

Usage:
    python src/models_roberta.py
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

MODEL_NAME   = 'roberta-base'
MAX_LEN      = 64
BATCH_SIZE   = 32
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.06
SEED         = 47

# feature index slices within the 8-column targetTitle feature block
COUNT_IDX = [0, 1]        # chars, words
STYLE_IDX = [4, 7, 6, 3]  # exclaim, uppercase_ratio, slang, adj
ALL_IDX   = slice(None)


def load_data():
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, 'X_train_features.csv'))
    X_val_df   = pd.read_csv(os.path.join(DATA_DIR, 'X_val_features.csv'))
    X_test_df  = pd.read_csv(os.path.join(DATA_DIR, 'X_test_features.csv'))
    y_train    = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze().astype(int).values
    y_val      = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).squeeze().astype(int).values
    y_test     = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze().astype(int).values

    # targetTitle feature block is the last 8 columns
    Xtr_tt = X_train_df.iloc[:, 8:].to_numpy(dtype=np.float32)
    Xvl_tt = X_val_df.iloc[:, 8:].to_numpy(dtype=np.float32)
    Xte_tt = X_test_df.iloc[:, 8:].to_numpy(dtype=np.float32)

    # raw titles, recreated with the same split used to build the feature CSVs
    train_inst  = pd.read_json(os.path.join(DATA_DIR, 'train', 'instances.jsonl'), lines=True)
    train_truth = pd.read_json(os.path.join(DATA_DIR, 'train', 'truth.jsonl'),    lines=True)
    test_inst   = pd.read_json(os.path.join(DATA_DIR, 'test',  'instances.jsonl'), lines=True)
    test_truth  = pd.read_json(os.path.join(DATA_DIR, 'test',  'truth.jsonl'),    lines=True)

    train_df = pd.merge(train_inst, train_truth, on='id')
    test_df  = pd.merge(test_inst,  test_truth,  on='id')
    train_df['targetTitle'] = train_df['targetTitle'].fillna('').astype(str)
    test_df['targetTitle']  = test_df['targetTitle'].fillna('').astype(str)

    train_split_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=SEED, stratify=train_df['truthClass']
    )

    titles_train = train_split_df['targetTitle'].tolist()
    titles_val   = val_df['targetTitle'].tolist()
    titles_test  = test_df['targetTitle'].tolist()

    assert len(titles_train) == len(y_train), 'Split mismatch — check random_state consistency'
    assert len(titles_val)   == len(y_val)
    assert len(titles_test)  == len(y_test)

    return (titles_train, titles_val, titles_test,
            Xtr_tt, Xvl_tt, Xte_tt,
            y_train, y_val, y_test)


def run_variant(case_name, feat_idx, titles_train, titles_val, titles_test,
                 Xtr_tt, Xvl_tt, Xte_tt, y_train, y_val, y_test, tokenizer, device):
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel, get_linear_schedule_with_warmup
    from torch.optim import AdamW

    class TitleFeatDataset(Dataset):
        def __init__(self, titles, feats, labels, idx):
            self.titles = titles
            self.labels = labels
            self.feats  = feats[:, idx] if idx is not None else None

        def __len__(self):
            return len(self.titles)

        def __getitem__(self, i):
            enc = tokenizer(self.titles[i], max_length=MAX_LEN, truncation=True,
                           padding='max_length', return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item['labels'] = torch.tensor(self.labels[i], dtype=torch.float)
            item['feats']  = torch.tensor(self.feats[i], dtype=torch.float) if self.feats is not None else None
            return item

    class RobertaHybrid(nn.Module):
        def __init__(self, feat_dim=0):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL_NAME)
            hidden = self.backbone.config.hidden_size
            if feat_dim > 0:
                self.feat_proj = nn.Sequential(
                    nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 32),
                    nn.ReLU(), nn.Dropout(0.1)
                )
                combined = hidden + 32
            else:
                self.feat_proj = None
                combined = hidden
            self.classifier = nn.Sequential(
                nn.Linear(combined, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
            )

        def forward(self, input_ids, attention_mask, feats=None):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            if self.feat_proj is not None and feats is not None:
                cls = torch.cat([cls, self.feat_proj(feats)], dim=1)
            return self.classifier(cls).squeeze(-1)

    tr_ds = TitleFeatDataset(titles_train, Xtr_tt, y_train, feat_idx)
    vl_ds = TitleFeatDataset(titles_val,   Xvl_tt, y_val,   feat_idx)
    te_ds = TitleFeatDataset(titles_test,  Xte_tt, y_test,  feat_idx)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE)

    feat_dim = 0 if feat_idx is None else Xtr_tt[:, feat_idx].shape[1]
    model = RobertaHybrid(feat_dim=feat_dim).to(device)

    steps = EPOCHS * math.ceil(len(tr_ds) / BATCH_SIZE)
    warmup = int(WARMUP_RATIO * steps)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps)
    loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(loader, train):
        model.train() if train else model.eval()
        total_loss, total, correct = 0.0, 0, 0
        preds_all, labels_all = [], []
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            feats = batch['feats'].to(device) if batch['feats'] is not None else None

            with torch.set_grad_enabled(train):
                logits = model(ids, mask, feats)
                loss = loss_fn(logits, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            preds_all.extend(preds.tolist())
            labels_all.extend(labels.cpu().numpy().astype(int).tolist())
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels.cpu().numpy().astype(int)).sum()
        return total_loss / total, correct / total, np.array(preds_all), np.array(labels_all)

    print(f'\n==================== {case_name} ====================')
    best_acc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(tr_loader, train=True)
        vl_loss, vl_acc, _, _ = run_epoch(vl_loader, train=False)
        print(f'  [Epoch {ep}] train_loss={tr_loss:.4f}  val_acc={vl_acc:.4f}')
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _, vl_acc, vl_preds, vl_labels = run_epoch(vl_loader, train=False)
    _, te_acc, te_preds, te_labels = run_epoch(te_loader, train=False)

    rep_val  = classification_report(vl_labels, vl_preds, output_dict=True, zero_division=0)
    rep_test = classification_report(te_labels, te_preds, output_dict=True, zero_division=0)

    print(f'  Val acc={vl_acc:.4f}  Test acc={te_acc:.4f}')

    return {
        'Variant':   case_name,
        'Val_Acc':   round(vl_acc, 4),
        'Val_F1':    round(rep_val['macro avg']['f1-score'], 4),
        'Test_Acc':  round(te_acc, 4),
        'Test_F1':   round(rep_test['macro avg']['f1-score'], 4),
    }


def main():
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print('PyTorch / transformers not installed. Install with:')
        print('  pip install torch transformers')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if device == 'cpu':
        print('WARNING: no GPU detected. RoBERTa fine-tuning will be very slow on CPU.')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print('Loading data...')
    (titles_train, titles_val, titles_test,
     Xtr_tt, Xvl_tt, Xte_tt, y_train, y_val, y_test) = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    variants = [
        ('RoBERTa (targetTitle only)',       None),
        ('RoBERTa + Count-based',            COUNT_IDX),
        ('RoBERTa + Style-based',            STYLE_IDX),
        ('RoBERTa + All features',           ALL_IDX),
    ]

    results = []
    for name, idx in variants:
        res = run_variant(name, idx, titles_train, titles_val, titles_test,
                          Xtr_tt, Xvl_tt, Xte_tt, y_train, y_val, y_test,
                          tokenizer, device)
        results.append(res)

    df = pd.DataFrame(results).sort_values('Test_F1', ascending=False)
    print('\n=== SUMMARY ===')
    print(df.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, 'roberta_results.csv'), index=False)
    print(f'\nSaved results to {RESULTS_DIR}/roberta_results.csv')


if __name__ == '__main__':
    main()
