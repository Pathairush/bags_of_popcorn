import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

from typing import List

from tqdm.notebook import tqdm

from tqdm.auto import tqdm
tqdm.pandas()

import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from keras.layers import Input, Embedding, Dropout, Conv1D, MaxPool1D, GRU, Dense
from keras.models import Model

def review_to_words(review:str, remove_stopwords:bool=False, return_string=False) -> List[str]:
    
    text = BeautifulSoup(review).get_text()
    letters = re.sub("[^a-zA-Z]", " ", text)
    words = letters.lower().split()
    
    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words('english')]
    
    if return_string:
        words = ' '.join(words)
    
    return words

def clean_review(df):
    
    df['review'] = df['review'].progress_apply(review_to_words, remove_stopwords = True, return_string=True)
    
    return df

def split_train_validation(df):
    
    X, X_val, y, y_val = train_test_split(df['review'], df['sentiment'], test_size = .2, random_state = 42)
    
    return X, X_val, y, y_val

def preprocessing_keras(df, tokenizer=None) -> list:
    
    NUM_MOST_FREQ_WORDS = 5_000
    MAX_REVIEW_LENGTH = 500
    
    if not tokenizer:
        
        # Train set
        df = clean_review(df)
        
        tokenizer = Tokenizer(num_words = NUM_MOST_FREQ_WORDS)
        tokenizer.fit_on_texts(df['review'].tolist())
        
        X, X_val, y, y_val = split_train_validation(df)
        X, X_val = X.tolist(), X_val.tolist()
        
        X, X_val = tokenizer.texts_to_sequences(X), tokenizer.texts_to_sequences(X_val)
        X, X_val = pad_sequences(X, maxlen = MAX_REVIEW_LENGTH), pad_sequences(X_val, maxlen = MAX_REVIEW_LENGTH)
        
        return X, X_val, y, y_val, tokenizer
    
    else:
        
        # Test set
        df = clean_review(df)
        X = df['review'].tolist()
        
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen = MAX_REVIEW_LENGTH)
        
        return X

def initial_model():
    
    NUM_MOST_FREQ_WORDS = 5_000
    MAX_REVIEW_LENGTH = 500
    EMBEDDING_VECTOR_LENGTH = 32
    
    inputs = Input(shape = (MAX_REVIEW_LENGTH,))
    embedding_layer = Embedding(input_dim = NUM_MOST_FREQ_WORDS, output_dim = EMBEDDING_VECTOR_LENGTH, input_length = MAX_REVIEW_LENGTH)
    X = embedding_layer(inputs)
    X = Dropout(.2)(X)
    X = Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')(X)
    X = MaxPool1D(pool_size = 2)(X)
    X = GRU(100, dropout = .0, recurrent_dropout = .0)(X)
    X = Dropout(.2)(X)
    outputs = Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(), metrics = [tf.keras.metrics.AUC()])
    
    return model

def create_early_stopping():
    
    '''
    This callback allows you to specify the performance measure to monitor, the trigger, and once triggered, it will stop the training process.
    '''
    
    return tf.keras.callbacks.EarlyStopping(monitor = 'val_loss')

def create_checkpoint():
    
    '''
    Checkpointing in Keras
    The EarlyStopping callback will stop training once triggered, 
    but the model at the end of training may not be the model with best performance on the validation dataset.

    An additional callback is required that will save the best model observed during training for later use. 
    This is the ModelCheckpoint callback.
    Reference : https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    '''
    
    return tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

def print_roc_auc_test_set():
    
    '''
    Read the prediction score and calculate the testing set score.
    We know the actual target label from the leakage issue from this discussion : https://www.kaggle.com/c/word2vec-nlp-tutorial/discussion/27022
    The purpose is to evaluate the model without submitting the csv to the system.
    '''

    y_score = pd.read_csv('/kaggle/working/submission.csv')['sentiment']

    test_file = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(test_file, compression='zip', header=0, delimiter='\t',quoting=3)
    y_true = test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)

    print( f"The AUC Score for testing set is : {roc_auc_score(y_true, y_score):.4f}" )

def main_dl():
    
    train_file = "/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
    train = pd.read_csv(train_file, compression='zip', header=0, delimiter='\t',quoting=3)
    
    print('--- Start preparing training set')
    X, X_val, y, y_val, tokenizer = preprocessing_keras(train)
    
    print('--- Start fitting model')
    gru_model = initial_model()
    es = create_early_stopping()
    cp = create_checkpoint()
    gru_model.fit(X, y, batch_size = 128, epochs = 5, validation_data = (X_val, y_val), callbacks=[es, cp])
    gru_model.load_weights('./checkpoint') # weight with minimum validataion loss
    
    print('--- Start preparing testing set')
    test_file = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(test_file, compression='zip', header=0, delimiter='\t',quoting=3)
    
    X_test = preprocessing_keras(test, tokenizer)
    pred = gru_model.predict(X_test)[:,0]
    
    output = pd.DataFrame( data = {"id" : test["id"], 'sentiment' : pred})
    output.to_csv("submission.csv", index=False, quoting=3)
    
    print_roc_auc_test_set()
    
    print('--- Done')
    
if __name__ == "__main__":
    
    main_dl()