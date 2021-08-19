from gensim.models import word2vec
import time
import logging
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from typing import List

from tqdm.notebook import tqdm

from tqdm.auto import tqdm
tqdm.pandas()

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def review_to_words(review: str, remove_stopwords: bool = False, return_string=False) -> List[str]:

    text = BeautifulSoup(review).get_text()
    letters = re.sub("[^a-zA-Z]", " ", text)
    words = letters.lower().split()

    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words('english')]

    if return_string:
        words = ' '.join(words)

    return words


def review_to_sentence(review: str, tokenizer: nltk.tokenize.punkt.PunktSentenceTokenizer, remove_stopwords: bool = False) -> List[List[str]]:

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


def create_word_centroid_map(w2v_model):

    print('--- Start fitting clustering model')
    start = time.time()

    word_vectors = w2v_model.wv.vectors
    num_clusters = word_vectors.shape[0] // 5

    cluster_model = KMeans(n_clusters=num_clusters)
    idx = cluster_model.fit_predict(word_vectors)

    end = time.time()
    print(f'--- Clustering process time : {end - start} seconds')

    word_centroid_map = dict(zip(w2v_model.wv.index_to_key, idx))

    return word_centroid_map


def create_bag_of_centroids(review, word_centroid_map):

    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    for word in review:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids


def main():

    w2v_model = train_w2v()
    num_clusters = w2v_model.wv.vectors.shape[0] // 5

    word_centroid_map = create_word_centroid_map(w2v_model)

    train_file = "/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
    train = pd.read_csv(train_file, compression='zip',
                        header=0, delimiter='\t', quoting=3)

    clean_train_reviews = []
    for review in tqdm(train["review"]):
        clean_train_reviews.append(
            review_to_words(review, remove_stopwords=True))

    X = np.zeros((train["review"].size, num_clusters), dtype="float32")
    counter = 0
    for review in tqdm(clean_train_reviews):
        X[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    y = train["sentiment"]

    print('--- Start fitting classification model')
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    test_file = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(test_file, compression='zip',
                       header=0, delimiter='\t', quoting=3)

    clean_test_reviews = []
    for review in tqdm(test["review"]):
        clean_test_reviews.append(
            review_to_words(review, remove_stopwords=True))

    test_X = np.zeros((test["review"].size, num_clusters), dtype="float32")
    counter = 0
    for review in tqdm(clean_test_reviews):
        test_X[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    pred = model.predict(test_X)
    output = pd.DataFrame(data={"id": test["id"], 'sentiment': pred})
    output.to_csv("submission.csv", index=False, quoting=3)


if __name__ == "__main__":

    main()
