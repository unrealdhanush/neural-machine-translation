import os
import argparse
import re
import numpy as np
import keras
from keras import regularizers
from keras.models import Model
from keras.layers import GRU, Dense, Embedding, Bidirectional, Input, Concatenate, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tensorflow as tf

@keras.utils.register_keras_serializable()
# Bahdanau Attention Layer
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Constants
MAX_VOCAB_SIZE = 10000  # Limiting vocabulary size
EMBEDDING_DIM = 256  # Reduced embedding dimension
GRU_UNITS = 256  # Reduced GRU units
BATCH_SIZE = 128  # Increased batch size for faster training
EPOCHS = 1  # Total epochs
DROPOUT_RATE = 0.3  # Reduced dropout rate

# Load and preprocess data
def load_data(filename):
    with open(filename, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

def preprocess_text(sentences):
    preprocessed = []
    for sentence in sentences:
        sentence = sentence.lower()
        # sentence = re.sub(r'[^a-zäöüß\s]', '', sentence)
        sentence = '<start> ' + sentence + ' <end>'
        preprocessed.append(sentence)
    return preprocessed

def tokenize_and_pad(sentences, max_len):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, char_level=False, filters='', lower=False)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return tokenizer, padded

def create_model(de_vocab_size, en_vocab_size, max_len):
    # Encoder
    enc_input = Input(shape=(max_len,))
    enc_emb = Embedding(input_dim=de_vocab_size, output_dim=EMBEDDING_DIM)(enc_input)
    enc_output, state_h = GRU(GRU_UNITS, return_state=True, return_sequences=True)(enc_emb)

   # Decoder
    dec_input = Input(shape=(max_len-1,))
    dec_emb_layer = Embedding(input_dim=en_vocab_size, output_dim=EMBEDDING_DIM)
    dec_emb = dec_emb_layer(dec_input)

    # Attention
    attention = BahdanauAttention(GRU_UNITS)
    context_vector, attention_weights = attention(state_h, enc_output)

    # Adjusted Context Vector Repetition
    context_vector = tf.expand_dims(context_vector, 1)
    context_vector = tf.repeat(context_vector, max_len, axis=1)

    dec_emb = dec_emb[:, :max_len, :]
    padding = tf.maximum(0, max_len - tf.shape(dec_emb)[1])
    dec_emb_padded = tf.pad(dec_emb, [[0, 0], [0, padding], [0, 0]])

    # Concatenating the context vector with the decoder input
    dec_input_combined = Concatenate(axis=-1)([context_vector, dec_emb_padded])

    # Decoder GRU - Ensure input shape is correct
    decoder_gru = GRU(GRU_UNITS, return_sequences=True, return_state=True)
    dec_output, _ = decoder_gru(dec_input_combined, initial_state=[state_h])

    # Dense layer
    dec_dense = Dense(en_vocab_size, activation='softmax')
    dec_output = dec_dense(dec_output)

    # Define the model
    model = Model([enc_input, dec_input], dec_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_target_data(en_train_pad):
    # Shift target output by one position, removing the first token
    target_data = np.roll(en_train_pad, -1, axis=1)
    # Set the last token of each sequence to 0 (assuming 0 is the padding index)
    target_data[:, -1] = 0
    return target_data

def train_model(model, de_train_pad, en_train_pad, target_data):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit([de_train_pad, en_train_pad[:, :-1]], target_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])    
    model_directory = 'model'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model.save(os.path.join(model_directory, 'nmt_model.keras'))

def translate(sentence, model, de_tokenizer, en_tokenizer, max_len):
    sentence = preprocess_text([sentence])[0]
    #print(sentence)
    sequence = de_tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    # Decoder input is a sequence of '<start>' tokens (here, represented by index 1)
    decoder_input = np.zeros((1, max_len - 1))  # Assuming index 1 represents '<start>'
    prediction = model.predict([sequence, decoder_input])[0]
    predicted_indices = np.argmax(prediction, axis=-1)
    print(predicted_indices)
    translated_sentence = []
    # print(predicted_indices)
    for idx in predicted_indices:
        word = en_tokenizer.index_word.get(idx, None)
        if word is None or word == '<end>':
            break
        translated_sentence.append(word)
    return translated_sentence

def test_model(model, de_test, en_test, de_tokenizer, en_tokenizer, max_len):
    translations = [translate(de_test[i], model, de_tokenizer, en_tokenizer, max_len) for i in range(len(de_test))]
    references = [[ref.split()] for ref in en_test]
    # print(references)
    print(translations)
    for t in translations:
        # t = ' '.join(t)
        print(type(t))
        bleu = sentence_bleu(references, t, smoothing_function=SmoothingFunction().method1)
        print('BLEU score:', bleu)

def main():
    parser = argparse.ArgumentParser(description='Neural Machine Translation Model')
    parser.add_argument('command', choices=['train', 'test'], help='Command to execute: train or test')
    args = parser.parse_args()

    # Load training and test data
    en_train_raw = load_data('data/train.envi.txt')
    de_train_raw = load_data('data/train.vi.txt')
    en_test_raw = load_data('data/tst2012.en.txt')
    de_test_raw = load_data('data/tst2012.vi.txt')

    # Preprocess data
    en_train = preprocess_text(en_train_raw)
    de_train = preprocess_text(de_train_raw)
    en_test = preprocess_text(en_test_raw)
    de_test = preprocess_text(de_test_raw)
    print(de_test[0])

    # Create tokenizers and fit on data
    max_len = 100
    de_tokenizer, de_train_pad = tokenize_and_pad(de_train_raw, max_len)
    en_tokenizer, en_train_pad = tokenize_and_pad(en_train_raw, max_len)

    print("Input data shape:", de_train_pad.shape)
    
    # Prepare target data
    target_data = create_target_data(en_train_pad)
    print("Target data shape:", target_data.shape)

    assert de_train_pad.shape[0] == target_data.shape[0], "Mismatch in number of samples between input and target data"
    
    # Get max length and vocab size
    de_vocab_size = len(de_tokenizer.word_index) + 1
    en_vocab_size = len(en_tokenizer.word_index) + 1

    # Create model
    model = create_model(de_vocab_size, en_vocab_size, max_len)

    if args.command == 'train':
        train_model(model, de_train_pad, en_train_pad, target_data)
    elif args.command == 'test':
        # Load the saved model
        model = keras.models.load_model('model/nmt_model.keras')
        test_model(model, de_test_raw, en_test_raw, de_tokenizer, en_tokenizer, max_len)

if __name__ == '__main__':
    main()
