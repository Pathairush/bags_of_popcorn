import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from typing import Tuple


def review_to_words(review: str) -> str:
    '''
    Cleaning the input review by the following steps
    1. Remove HTML tags
    2. Replace non-letter characters with white space
    3. Lower case all characters
    4. Remove stopwords based on NLTK library
    '''

    remove_html_tag = BeautifulSoup(review)
    letter_only = re.sub("[^a-zA-Z]", " ", remove_html_tag.get_text())
    lower_cases = letter_only.lower()
    words = lower_cases.split(' ')
    words = [w for w in words if w not in stopwords.words('english')]

    return ' '.join(words)


def preprocess_data(df: pd.DataFrame, vectorizer: CountVectorizer = None) -> Tuple[np.array, pd.Series, CountVectorizer]:
    '''
    Pre-process data by applying the review to words function to each sample.
    Also, transform the cleaned review sentences into bag of words features.
    '''

    df['cleaned_review'] = df['review'].progress_apply(review_to_words)

    if vectorizer is None:
        vectorizer = CountVectorizer(
            analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
        X = vectorizer.fit_transform(df['cleaned_review']).toarray()
    else:
        X = vectorizer.transform(df['cleaned_review']).toarray()

    if "sentiment" in df.columns:
        y = df["sentiment"]
    else:
        y = None

    return X, y, vectorizer


def main():

    train_file = "/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
    train = pd.read_csv(train_file, compression='zip',
                        header=0, delimiter='\t', quoting=3)

    X, y, vectorizer = preprocess_data(train)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)

    test_file = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
    test = pd.read_csv(test_file, compression='zip',
                       header=0, delimiter='\t', quoting=3)

    X, _, _ = preprocess_data(test, vectorizer)
    pred = model.predict(X)
    output = pd.DataFrame(data={"id": test["id"], 'sentiment': pred})
    output.to_csv("submission.csv", index=False, quoting=3)


if __name__ == "__main__":

    main()
