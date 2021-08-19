import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec

from sklearn.ensemble import RandomForestClassifier

from typing import Tuple, List

from tqdm.notebook import tqdm

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def review_to_words(review: str, remove_stopwords: bool = False) -> List[str]:
    '''
    Cleaning the input review by the following steps
    1. Remove HTML tags
    2. Replace non-letter characters with white space
    3. Lower case all characters
    4. Remove stopwords based on NLTK library
    '''

    text = BeautifulSoup(review).get_text()
    letters = re.sub("[^a-zA-Z]", " ", text)
    words = letters.lower().split()

    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words('english')]

    return words


def review_to_sentence(review: str, tokenizer: nltk.tokenize.punkt.PunktSentenceTokenizer, remove_stopwords: bool = False) -> List[List[str]]:
    '''
    Cleaning each sentence in review
    '''
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_words(raw_sentence, remove_stopwords))

    return sentences


def train_w2v():

    train_file = "/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
    train = pd.read_csv(train_file, compression='zip',
                        header=0, delimiter='\t', quoting=3)

    test_file = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(test_file, compression='zip',
                       header=0, delimiter='\t', quoting=3)

    sentences = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for review in tqdm(train['review']):
        sentences += review_to_sentence(review, tokenizer)

    for review in tqdm(test['review']):
        sentences += review_to_sentence(review, tokenizer)

    print(f'How many sentences we have in total? : {len(sentences)} sentences')

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features,
                              min_count=min_word_count, window=context, sample=downsampling)

    # On a dual-core Macbook Pro, this took less than 15 minutes to run using 4 worker threads.\
    model_name = '300feats_40minwords_10context.model'
    print('--- Start fitting word2vec model')
    model.save(model_name)

    print(f''' --- Let's see what we get from Word2Vec --- Doesn't Match ''')
    print(
        f'''Which word doesn't match? [man woman child kitchen] : {model.wv.doesnt_match("man woman child kitchen".split())}''')
    print(
        f'''Which word doesn't match? [france england germany berlin] : {model.wv.doesnt_match("france england germany berlin".split())}''')
    print(
        f'''Which word doesn't match? [paris berlin london austria] : {model.wv.doesnt_match("paris berlin london austria".split())}''')

    print(f''' --- Let's see what we get from Word2Vec --- Similarity ''')
    print(
        f''' Which words are most similar to man : {model.wv.most_similar("man")} ''')

    return model


def make_feature_vector(review: list, w2v_model, num_features: int) -> np.array:
    '''
    Averaging each word feature value in 1 review together (sum all feature values and divide number of words existed in model vocabulary).
    Return the numpy array in size of (num_features,)
    '''

    feature_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0
    ind2word_set = set(w2v_model.wv.index_to_key)

    for word in review:
        if word in ind2word_set:
            nwords += 1
            feature_vector = np.add(feature_vector, w2v_model.wv[word])

    feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def get_avg_feature_vector(reviews: list, w2v_model, num_features: int) -> np.array:
    '''
    For each review, get a feature vector based on the average of each word in the review.
    Return the numpy array in size of (len(reviews), num_features)
    '''

    print('--- Get average feature vectors')
    review_feature_vector = np.zeros(
        (len(reviews), num_features), dtype="float32")
    counter = 0

    for review in tqdm(reviews):
        review_feature_vector[counter] = make_feature_vector(
            review, w2v_model, num_features)
        counter += 1

    return review_feature_vector


def main():

    w2v_model = train_w2v()

    train_file = "/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
    train = pd.read_csv(train_file, compression='zip',
                        header=0, delimiter='\t', quoting=3)

    clean_train_reviews = []
    for review in tqdm(train['review']):
        clean_train_reviews.append(
            review_to_words(review, remove_stopwords=True))

    X = get_avg_feature_vector(
        clean_train_reviews, w2v_model, num_features=300)
    y = train["sentiment"]

    print('--- Start fitting classification model')
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    test_file = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(test_file, compression='zip',
                       header=0, delimiter='\t', quoting=3)

    clean_test_reviews = []
    for review in tqdm(test['review']):
        clean_test_reviews.append(
            review_to_words(review, remove_stopwords=True))

    test_X = get_avg_feature_vector(
        clean_test_reviews, w2v_model, num_features=300)

    pred = model.predict(test_X)
    output = pd.DataFrame(data={"id": test["id"], 'sentiment': pred})
    output.to_csv("submission.csv", index=False, quoting=3)


if __name__ == "__main__":

    main()
