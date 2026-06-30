"""
models_deep.py
---------------
Trains a 1D Convolutional Neural Network on raw targetTitle text
to capture local n-gram patterns that handcrafted features and
TF-IDF cannot.

Architecture:
    Embedding -> Conv1D -> GlobalMaxPool -> Dense -> Dropout -> Sigmoid

Usage:
    python src/models_deep.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

RANDOM_STATE = 47
MAX_TOKENS   = 30000
MAX_LEN      = 40       # headlines are short
BATCH_SIZE   = 64
EPOCHS       = 6


def load_raw_corpus():
    train_inst  = pd.read_json(os.path.join(DATA_DIR, 'train', 'instances.jsonl'), lines=True)
    train_truth = pd.read_json(os.path.join(DATA_DIR, 'train', 'truth.jsonl'),    lines=True)
    test_inst   = pd.read_json(os.path.join(DATA_DIR, 'test',  'instances.jsonl'), lines=True)
    test_truth  = pd.read_json(os.path.join(DATA_DIR, 'test',  'truth.jsonl'),    lines=True)

    train_df = pd.merge(train_inst, train_truth, on='id')
    test_df  = pd.merge(test_inst,  test_truth,  on='id')

    for df in [train_df, test_df]:
        df['targetTitle'] = df['targetTitle'].fillna('').astype(str)
        df['truthClass']  = df['truthClass'].apply(lambda x: 1 if x == 'clickbait' else 0)

    return train_df, test_df


def build_cnn(text_vec):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Embedding(input_dim=MAX_TOKENS, output_dim=128, mask_zero=True),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print('TensorFlow not installed. Install with: pip install tensorflow')
        return

    print('Loading corpus...')
    train_df, test_df = load_raw_corpus()

    train_split_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=train_df['truthClass']
    )

    X_train = train_split_df['targetTitle'].tolist()
    y_train = train_split_df['truthClass'].values
    X_val   = val_df['targetTitle'].tolist()
    y_val   = val_df['truthClass'].values
    X_test  = test_df['targetTitle'].tolist()
    y_test  = test_df['truthClass'].values

    print(f'Sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}')

    text_vec = layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_mode='int', output_sequence_length=MAX_LEN,
        standardize='lower_and_strip_punctuation', split='whitespace'
    )
    text_vec.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(512))

    def make_ds(texts, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        ds = ds.map(lambda x, y: (text_vec(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(4096, seed=RANDOM_STATE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_train = make_ds(np.array(X_train), y_train, training=True)
    ds_val   = make_ds(np.array(X_val),   y_val)
    ds_test  = make_ds(np.array(X_test),  y_test)

    model = build_cnn(text_vec)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor='val_accuracy')
    ]

    print('\nTraining CNN...')
    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    def predict(ds):
        probs = model.predict(ds, verbose=0).ravel()
        return (probs >= 0.5).astype(int)

    yv_pred = predict(ds_val)
    yt_pred = predict(ds_test)

    rep_val  = classification_report(y_val,  yv_pred, output_dict=True, zero_division=0)
    rep_test = classification_report(y_test, yt_pred, output_dict=True, zero_division=0)
    cm_test  = confusion_matrix(y_test, yt_pred)

    print('\n=== CNN Validation ===')
    print(classification_report(y_val, yv_pred, digits=4))
    print('\n=== CNN Test ===')
    print(classification_report(y_test, yt_pred, digits=4))
    print('\nConfusion Matrix (Test):\n', cm_test)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = pd.DataFrame([
        {'Tag': 'CNN | VAL',  'Accuracy': rep_val['accuracy'],
         'Precision': rep_val['macro avg']['precision'],
         'Recall': rep_val['macro avg']['recall'],
         'F1': rep_val['macro avg']['f1-score']},
        {'Tag': 'CNN | TEST', 'Accuracy': rep_test['accuracy'],
         'Precision': rep_test['macro avg']['precision'],
         'Recall': rep_test['macro avg']['recall'],
         'F1': rep_test['macro avg']['f1-score']},
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, 'cnn_results.csv'), index=False)
    print(f'\nSaved results to {RESULTS_DIR}/cnn_results.csv')


if __name__ == '__main__':
    main()
