import numpy as np
import re  # regex - regular expressions
import pickle


def process_tweet(tweet: str) -> np.ndarray:
    """
    Preprocess the tweet to lowercase, get rid of punctuation marks and replace:

        URLs -> httpaddr

        numbers -> numberrep 

        @ -> tagrep 

    Parameters
    ----------
    tweet : string
        tweet to preprocess

    Returns
    -------
    result : ndarray
        A numpy ndarray of words
    """

    result = tweet

    # lower case
    result = result.lower()

    # replace URLs with ' httpaddr '
    result = re.compile('(http|https)://[^\s]*').sub('', result)

    # replace numbers with ' number '
    result = re.compile('[0-9]+').sub('', result)

    # replacing tags with ' tagrep '
    result = re.compile('[@]+').sub('', result)

    # get rid of punctuation
    result = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', result)

    # remove empty strings
    result = [word for word in result if len(word) > 0]

    return np.array(result)


def load_dictionary(directory) -> tuple:
    word_to_vector = pickle.load(open(f'{directory}word_to_vector.pkl', 'rb'))
    words_to_index = pickle.load(open(f'{directory}words_to_index.pkl', 'rb'))
    index_to_words = pickle.load(open(f'{directory}index_to_words.pkl', 'rb'))
    return word_to_vector, words_to_index, index_to_words


def load_training_input() -> tuple:
    training_input = pickle.load(open('../data/training_input.pkl', 'rb'))
    X, y = training_input['X'], training_input['y']
    return X, y
