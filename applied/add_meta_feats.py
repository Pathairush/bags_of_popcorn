from collections import defaultdict, Counter
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Conv1D, MaxPool1D, LSTM, Dense, Bidirectional, Concatenate
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import logging
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

from typing import List

from tqdm.notebook import tqdm

from tqdm.auto import tqdm
tqdm.pandas()

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def review_to_words(review: str, remove_stopwords: bool = False, return_string=False) -> List[str]:
    '''
    Clean the review with the following methods
    1. Remove HTML markdown
    2. Filter only letter only
    3. Lower case all the characters
    4. (optional) remove stopwords

    Args:
    review -> review string
    remove_stopwords -> option whether the funciton remove the stopwords
    return_string -> option on whether the function return 'list of words' or 'string'

    Returns:
    words -> string, or list of string based on the `return_string` argument.
    '''

    text = BeautifulSoup(review).get_text()
    letters = re.sub("[^a-zA-Z]", " ", text)
    words = letters.lower().split()

    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words('english')]

    if return_string:
        words = ' '.join(words)

    return words


def clean_review(df):
    '''
    Apply the review to words function to review in dataset.

    Args:
    df -> train or test dataframe

    Returns:
    df -> dataframe with cleaned review
    '''

    df['review'] = df['review'].progress_apply(
        review_to_words, remove_stopwords=True, return_string=True)

    return df


def split_train_validation(df):
    '''
    Split the train dataframe into train, and test set with 80:20 propotion.

    Args:
    df -> train dataframe

    Returns:
    X, X_val -> train dataframe, validated dataframe
    y, y_val -> train target, validated target
    '''

    X, X_val, y, y_val = train_test_split(df, df['sentiment'],
                                          test_size=.2, random_state=42)

    return X, X_val, y, y_val


def calculate_num_most_frequent_word(train, coverage=.8):
    '''
    Caculate number of the most frequent words to use in word embedding based on the provided percent coverage of total words.

    Args:
    train -> train dataframe with `review` column.
    coverage -> the percent coverage of the most frequent word to the total words.

    Returns:
    num_most_freq_word -> the number of unique words that cover the provided coverage of the total words.
    '''

    train_words = train['review'].str.split().tolist()
    uniq_words = set()
    words = list()
    for word in train_words:
        uniq_words.update(set(word))
        words += word

    num_occur = Counter(words)
    print(
        f'There are {len(uniq_words):,} unique words. The total number of word is : {sum(num_occur.values()):,}')

    count = 0
    num_most_freq_word = 0
    for num in sorted(num_occur.values(), reverse=True):
        count += num
        num_most_freq_word += 1
        if count > sum(num_occur.values()) * coverage:
            break

    print(
        f'To cover {coverage * 100}% of total words, we can use top {num_most_freq_word:,} most frequent word.')

    return num_most_freq_word


def calculate_max_review_lenght(train):
    '''
    Calculate the max review length of training data set to be used with the pad_sequences preprocessing function.

    Args:
    train -> The train dataframe with `review` columns.

    Returns:
    max_review_length -> the max size of review length based on the training dataframe.
    '''

    max_review_length = max(
        train['review'].str.split().apply(lambda x: len(x)))
    print(f'The max review lenght from train set is : {max_review_length:,}')

    return max_review_length


def preprocess_data(df, num_most_freq_word, max_review_length, tokenizer=None, meta_features=None, scaler=None) -> list:
    '''
    Convert text to sequences based on the `num_most_freq_word`
    And, pad them with the `max_review_lenght` argument.
    We seperate the behavior of the funciton (between train and test) based on whether the tokenzier is provided or not.

    Args:
    df -> train or test dataframe.
    num_most_freq_word -> the number of unique words that cover the provided coverage of the total words.
    max_review_length -> the max size of review length based on the training dataframe.
    tokenizer -> if it's none, fit the tokenizer based on the provided `df`. else, use the provided tokenizer for preprocessing data.
    scaler -> if it's nont, fit the standard scaler based on the splitted training data set.

    Returns:
    If tokenizer provided
    X, X_val -> train sequences, validated sequences
    y, y_val -> train target, validated target
    tokenizer -> the tokenizer that fits on all train data.
    scaler -> the standard scaler that fits on splitted train data.

    else
    X -> test sequences
    '''

    if not tokenizer:  # train set

        # fit the tokenizer based on data in training set
        tokenizer = Tokenizer(num_words=num_most_freq_word)
        tokenizer.fit_on_texts(df['review'].tolist())

        # split train, validate dataframe
        X, X_val, y, y_val = split_train_validation(df)

        if meta_features:

            print(
                f'-- preprocess_data (train) : meta_feature is activated : {meta_featurs}')
            X_meta, X_val_meta = X[meta_features], X_val[meta_features]

            scaler = StandardScaler()
            scaler.fit(X_meta)
            X_meta, X_val_meta = scaler.transform(
                X_meta), scaler.transform(X_val_meta)

            X, X_val = X['review'], X_val['review']

        X, X_val = X.tolist(), X_val.tolist()

        # convert to sequences and padding them to be the same lenght
        X, X_val = tokenizer.texts_to_sequences(
            X), tokenizer.texts_to_sequences(X_val)
        X, X_val = pad_sequences(X, maxlen=max_review_length), pad_sequences(
            X_val, maxlen=max_review_lenght)

        if meta_features:

            print(
                f'-- preprocess_data (test) : meta_feature is activated : {meta_featurs}')
            return X, X_val, y, y_val, tokenizer, X_meta, X_val_meta, scaler

        return X, X_val, y, y_val, tokenizer

    else:  # test set

        # processing the test set based on the provided trained tokenizer
        X = df['review'].tolist()
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=max_review_length)

        if meta_features:

            X_meta = df[meta_features]
            X_meta = scaler.transform(X_meta)

            return X, X_meta

        return X


def feature_engineering(df):
    '''
    Extract the feature from the review.
    The idea is inspired by : https://towardsdatascience.com/how-i-improved-my-text-classification-model-with-feature-engineering-98fbe6c13ef3

    Args:
    df -> train or test before cleaned the review dataframe.

    Returns:
    df -> Added featuers dataframe.
    '''

    df['word_count'] = df['review'].apply(lambda x: len(x.split()))
    df['char_count'] = df['review'].apply(lambda x: len(x.replace(" ", "")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    df['total_length'] = df['review'].apply(len)
    df['capitals'] = df['review'].apply(
        lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(
        row['capitals'])/float(row['total_length']), axis=1)
    df['num_exclamation_marks'] = df['review'].apply(lambda x: x.count('!'))
    df['num_question_marks'] = df['review'].apply(lambda x: x.count('?'))
    df['num_punctuation'] = df['review'].apply(
        lambda x: sum(x.count(w) for w in '.,;:'))
    df['num_symbols'] = df['review'].apply(
        lambda x: sum(x.count(w) for w in '*&$%'))
    df['num_unique_words'] = df['review'].apply(
        lambda x: len(set(w for w in x.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['word_count']
    df["word_unique_percent"] = df["num_unique_words"]*100/df['word_count']

    return df


def initial_model(num_most_freq_word, max_review_length, number_of_features):
    '''
    Initial the deep learning model basded on the provided input arguments

    Args:
    num_most_freq_word -> the number of unique words that cover the provided coverage of the total words.
    max_review_length -> the max size of review length based on the training dataframe.

    Returns:
    model -> keras model
    '''

    EMBEDDING_VECTOR_LENGTH = 128
    DROPOUT_RATIO = .2

    word_feats = Input(shape=(max_review_length,))

    X = Embedding(input_dim=num_most_freq_word,
                  output_dim=EMBEDDING_VECTOR_LENGTH,
                  input_length=max_review_length)(word_feats)
    X = Dropout(DROPOUT_RATIO)(X)
    X = Bidirectional(LSTM(100, dropout=.0, recurrent_dropout=.0))(X)
    stat_feats = Input(shape=(number_of_features,))
    X = Concatenate()([X, stat_feats])
    X = Dense(10)(X)
    X = Dropout(DROPOUT_RATIO)(X)

    outputs = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=[word_feats, stat_feats], outputs=outputs)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])

    return model


def create_early_stopping():
    '''
    This callback allows you to specify the performance measure to monitor, the trigger, and once triggered, it will stop the training process.

    Returns:
    kerase early stopping callback.
    '''

    return tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


def create_checkpoint():
    '''
    Checkpointing in Keras
    The EarlyStopping callback will stop training once triggered, 
    but the model at the end of training may not be the model with best performance on the validation dataset.

    An additional callback is required that will save the best model observed during training for later use. 
    This is the ModelCheckpoint callback.
    Reference : https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

    Returns:
    keras model checkpoint callback.
    '''

    return tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)


def print_roc_auc_test_set(test):
    '''
    Read the prediction score and calculate the testing set score.
    We know the actual target label from the leakage issue from this discussion : https://www.kaggle.com/c/word2vec-nlp-tutorial/discussion/27022
    The purpose is to evaluate the model without submitting the csv to the system.

    Args:
    test -> test dataset to extract the actual label for evaluation

    Returns:
    None
    '''

    y_score = pd.read_csv('/kaggle/working/submission.csv')['sentiment']

    y_true = test["id"].map(lambda x: 1 if int(
        x.strip('"').split("_")[1]) >= 5 else 0)

    print(
        f"The AUC Score for testing set is : {roc_auc_score(y_true, y_score):.4f}")


if __name__ == "__main__":

    train_file = "/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
    train = pd.read_csv(train_file, compression='zip',
                        header=0, delimiter='\t', quoting=3)

    train = feature_engineering(train)
    train = clean_review(train)

    test_file = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(test_file, compression='zip',
                       header=0, delimiter='\t', quoting=3)

    test = feature_engineering(test)
    test = clean_review(test)

    y_true = test["id"].map(lambda x: 1 if int(
        x.strip('"').split("_")[1]) >= 5 else 0)

    num_most_freq_word = calculate_num_most_frequent_word(train, coverage=.8)
    max_review_lenght = calculate_max_review_lenght(train)
    meta_features = [e for e in train.columns if e not in [
        'id', 'sentiment', 'review']]

    print('--- Start preprocessing data')

    X, X_val, y, y_val, tokenizer, X_meta, X_val_meta, scaler = preprocess_data(train,
                                                                                num_most_freq_word,
                                                                                max_review_lenght,
                                                                                tokenizer=None,
                                                                                meta_features=meta_features,
                                                                                scaler=None)

    print('--- Start fitting')

    model = initial_model(num_most_freq_word,
                          max_review_lenght, len(meta_features))
    es = create_early_stopping()
    cp = create_checkpoint()
    model.fit([X, X_meta], y, batch_size=64, epochs=10,
              validation_data=([X_val, X_val_meta], y_val), callbacks=[es, cp])
    model.load_weights('./checkpoint')

    print('--- Start predicting')

    X_test, X_test_meta = preprocess_data(test,
                                          num_most_freq_word,
                                          max_review_lenght,
                                          tokenizer,
                                          meta_features=meta_features,
                                          scaler=scaler)
    pred = model.predict([X_test, X_test_meta])[:, 0]

    output = pd.DataFrame(data={"id": test["id"], 'sentiment': pred})
    output.to_csv(f"submission.csv", index=False, quoting=3)

    print_roc_auc_test_set(test)

    print('--- Done')
