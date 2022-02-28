import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import load_dictionary

MAX_TWEET_LENGTH = 30


def load_model():
    print('\nLoading a model... It may take a second...\n')

    model = tf.keras.models.load_model('models/model1.h5')
    _, words_to_index, _ = load_dictionary('./data/dictionary/')
    tokenizer = Tokenizer()

    return model, words_to_index, tokenizer


def process_tweet(tweet, words_to_index, tokenizer):
    tokenizer.word_index = words_to_index

    tweet = tokenizer.texts_to_sequences([tweet])
    tweet = pad_sequences(tweet,
                          padding='post',
                          truncating='post',
                          maxlen=MAX_TWEET_LENGTH)

    return tweet


def main():
    model, words_to_index, tokenizer = load_model()

    print('\nPaste a tweet: ')
    tweet = input()

    processed_tweet = process_tweet(tweet, words_to_index, tokenizer)
    prediction = model.predict(processed_tweet)

    print(f"\nProbability that it's a disaster tweet: {float(np.squeeze(prediction))}\n")


if __name__ == '__main__':
    main()
