import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
# for more detail : https://radimrehurek.com/gensim/models/word2vec.html
from gensim.models import word2vec

from typing import List

from tqdm.notebook import tqdm


def review_to_words(review: str, remove_stopwords: bool = False) -> List[str]:

    text = BeautifulSoup(review).get_text()
    letters = re.sub("[^a-zA-Z]", " ", text)
    words = letters.lower().split()

    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words('english')]

    return words


def review_to_sentence(review: str, tokenizer: nltk.tokenize.punkt.PunktSentenceTokenizer, remove_stopwords: bool = False) -> List[List[str]]:

    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_words(raw_sentence, remove_stopwords))

    return sentences


def doesnt_match(words_list, model):
    return model.wv.doesnt_match(words_list)


def main():

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
    model.save(model_name)

    print(f'''--- Let's see what we get from Word2Vec --- Doesn't Match ''')
    print(
        f'''Which word doesn't match? [man woman child kitchen] : 
        {doesnt_match(['man', 'woman', 'child', 'kitchen'], model)}''')
    print(
        f'''Which word doesn't match? [france england germany berlin] : 
        {doesnt_match(['france', 'england', 'germany', 'berlin'], model)}''')
    print(
        f'''Which word doesn't match? [paris berlin london austria] : 
        {doesnt_match(['paris', 'berlin', 'london', 'austria'], model)}''')

    print(f'''--- Let's see what we get from Word2Vec --- Similarity ''')
    print(
        f'''Which words are most similar to man : 
        {model.wv.most_similar("man")} ''')


if __name__ == "__main__":

    main()
