"""
preprocessing.py
----------------
Loads the Webis Clickbait Corpus 2017, cleans text fields,
extracts handcrafted linguistic features, and saves the
feature matrices used by all downstream models.

Expected directory layout:
    data/
        train/instances.jsonl
        train/truth.jsonl
        test/instances.jsonl
        test/truth.jsonl
        all_slangs.csv

Outputs (written to data/):
    X_train_features.csv
    X_val_features.csv
    X_test_features.csv
    y_train.csv
    y_val.csv
    y_test.csv
"""

import re
import os
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAIN_INST  = os.path.join(DATA_DIR, 'train', 'instances.jsonl')
TRAIN_TRUTH = os.path.join(DATA_DIR, 'train', 'truth.jsonl')
TEST_INST   = os.path.join(DATA_DIR, 'test',  'instances.jsonl')
TEST_TRUTH  = os.path.join(DATA_DIR, 'test',  'truth.jsonl')
SLANG_CSV   = os.path.join(DATA_DIR, 'all_slangs.csv')

RANDOM_STATE = 47
VAL_SIZE     = 0.2

# ── feature names (order matches extract_features) ─────────────────────────
FEATURE_NAMES = [
    'chars', 'words', 'begins_q',
    'adj_count', 'exclaim_count', 'question_count',
    'slang_count', 'uppercase_ratio'
]


def load_corpus(inst_path, truth_path):
    inst  = pd.read_json(inst_path,  lines=True)
    truth = pd.read_json(truth_path, lines=True)
    df = pd.merge(inst, truth, on='id')
    df.drop(columns=['postMedia', 'postTimestamp'], errors='ignore', inplace=True)
    df['postText']    = df['postText'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    ).fillna('').astype(str).str.lower()
    df['targetTitle'] = df['targetTitle'].fillna('').astype(str).str.lower()
    df['truthClass']  = df['truthClass'].apply(lambda x: 1 if x == 'clickbait' else 0)
    return df


def load_slang_set(path):
    slang_df = pd.read_csv(path, header=None)
    return set(slang_df[0].astype(str).str.lower())


def extract_features(text, nlp, slang_set, question_words):
    """Return an 8-element feature vector for a single headline string."""
    text = text if isinstance(text, str) else ''
    tokens = text.split()

    char_count   = len(text)
    word_count   = len(tokens)
    begins_q     = int(bool(tokens) and tokens[0] in question_words)

    doc          = nlp(text)
    adj_count    = sum(1 for tok in doc if tok.pos_ == 'ADJ')

    excl_count   = text.count('!')
    quest_count  = text.count('?')
    slang_count  = sum(1 for tok in tokens if tok.lower() in slang_set)

    letters      = re.findall(r'[A-Za-z]', text)
    up_ratio     = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0

    return [char_count, word_count, begins_q, adj_count,
            excl_count, quest_count, slang_count, up_ratio]


def build_feature_matrix(df, fields, nlp, slang_set, question_words, label):
    """
    Extract features from each field in `fields` and concatenate horizontally.
    Column names are prefixed with the field name, e.g. postText_chars.
    """
    tqdm.pandas(desc=f'Extracting features ({label})')
    all_cols, all_arrays = [], []

    for field in fields:
        col_names = [f'{field}_{n}' for n in FEATURE_NAMES]
        arr = np.array(
            df[field].progress_apply(
                lambda t: extract_features(t, nlp, slang_set, question_words)
            ).tolist(),
            dtype=float
        )
        all_cols.extend(col_names)
        all_arrays.append(arr)

    return pd.DataFrame(np.hstack(all_arrays), columns=all_cols)


def main():
    print('Loading corpus...')
    train_df = load_corpus(TRAIN_INST, TRAIN_TRUTH)
    test_df  = load_corpus(TEST_INST,  TEST_TRUTH)

    print(f'  Train: {len(train_df)} | Test: {len(test_df)}')
    print(f'  Train class balance:\n{train_df["truthClass"].value_counts()}')

    train_split_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=train_df['truthClass']
    )
    print(f'  Train split: {len(train_split_df)} | Val: {len(val_df)}')

    print('Loading spaCy and slang dictionary...')
    nlp           = spacy.load('en_core_web_sm')
    slang_set     = load_slang_set(SLANG_CSV)
    question_words = {'what', 'who', 'when', 'why', 'where', 'how'}

    fields = ['postText', 'targetTitle']

    X_train = build_feature_matrix(train_split_df, fields, nlp, slang_set, question_words, 'train')
    X_val   = build_feature_matrix(val_df,          fields, nlp, slang_set, question_words, 'val')
    X_test  = build_feature_matrix(test_df,          fields, nlp, slang_set, question_words, 'test')

    y_train = train_split_df['truthClass'].values
    y_val   = val_df['truthClass'].values
    y_test  = test_df['truthClass'].values

    # save
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train_features.csv'), index=False)
    X_val.to_csv(  os.path.join(DATA_DIR, 'X_val_features.csv'),   index=False)
    X_test.to_csv( os.path.join(DATA_DIR, 'X_test_features.csv'),  index=False)
    pd.Series(y_train).to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    pd.Series(y_val).to_csv(  os.path.join(DATA_DIR, 'y_val.csv'),   index=False)
    pd.Series(y_test).to_csv( os.path.join(DATA_DIR, 'y_test.csv'),  index=False)

    print(f'\nSaved feature CSVs to {DATA_DIR}/')
    print(f'Feature shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')


if __name__ == '__main__':
    main()
