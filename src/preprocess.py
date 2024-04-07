import re
import os
import io
import time
import argparse
import numpy as np
import unicodedata
import keras
from sklearn.model_selection import train_test_split


def tokenize(lang):
    '''
        From TensorFlow NMT tutorial
    '''    

    lang_tokenizer = keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    '''
        From TensorFlow NMT tutorial
    '''

    # creating cleaned input, output pairs
    lang = create_dataset(path, num_examples)

    tensor, lang_tokenizer = tokenize(lang)

    return tensor, lang_tokenizer


def unicode_to_ascii(s):
    '''
        From TensorFlow NMT tutorial
    '''
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
    

def preprocess_sentence(w):
    '''
        From TensorFlow NMT tutorial
    '''
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples=None):
    '''
        From TensorFlow NMT tutorial
    '''
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    sentence_list = [preprocess_sentence(l) for l in lines[:num_examples]]
    return sentence_list


def preprocess(data_path, num_examples):
    '''
        train.envi.txt
        train.vi.txt
    '''
    train_en = os.path.join(data_path, 'train.envi.txt')
    train_vi = os.path.join(data_path, 'train.vi.txt')
     
    input_tensor, inp_lang = load_dataset(train_en, num_examples)   
    target_tensor, target_lang = load_dataset(train_vi, num_examples)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    return input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, inp_lang, target_lang    

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/')
    parser.add_argument('--num_examples', type=int, default=30000)
    args = parser.parse_args()
    preprocess(args.data_path, args.num_examples)

