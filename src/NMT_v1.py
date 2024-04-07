import os
import argparse
import re
import numpy as np
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import GRU, Dense, Embedding, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Constants
MAX_VOCAB_SIZE = 10000  # Limiting vocabulary size
EMBEDDING_DIM = 256  # Reduced embedding dimension
GRU_UNITS = 256  # Reduced GRU units
BATCH_SIZE = 128  # Increased batch size for faster training
EPOCHS = 10  # Total epochs
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
        sentence = re.sub(r'[^a-zäöüß\s]', '', sentence)
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
    model = Sequential()
    model.add(Embedding(input_dim=de_vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len))
    model.add(Bidirectional(GRU(GRU_UNITS, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))
    model.add(Dense(en_vocab_size, activation='softmax'))
    adam_optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_target_data(en_train_pad):
    target_data = np.zeros_like(en_train_pad)
    target_data[:,:-1] = en_train_pad[:,1:]
    return target_data

def train_model(model, de_train_pad, target_data):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(de_train_pad, target_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])
    # Ensure the directory exists
    model_directory = 'model'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # Save the model in the recommended format
    model.save(os.path.join(model_directory, 'nmt_model.keras'))

def translate(sentence, model, de_tokenizer, en_tokenizer, max_len):
    sentence = preprocess_text([sentence])[0]
    sequence = de_tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(sequence, verbose=0)[0]
    translated_sentence = []
    for idx in prediction.argmax(axis=-1):
        word = en_tokenizer.index_word.get(idx, None)
        if word is None or word == '<end>':
            break
        translated_sentence.append(word)
    return ' '.join(translated_sentence)

def test_model(model, de_test, en_test, de_tokenizer, en_tokenizer, max_len):
    translations = [translate(de_test[i], model, de_tokenizer, en_tokenizer, max_len) for i in range(len(de_test))]
    references = [[ref.split()] for ref in en_test]
    formatted_translations = [translation.split() for translation in translations]
    for t in translations:
        print(t)
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

    # Create tokenizers and fit on data
    max_len = 100
    de_tokenizer, de_train_pad = tokenize_and_pad(de_train, max_len)
    en_tokenizer, en_train_pad = tokenize_and_pad(en_train, max_len)

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
        train_model(model, de_train_pad, target_data)
    elif args.command == 'test':
        # Load the saved model
        model = keras.models.load_model('model/nmt_model.keras')
        test_model(model, de_test, en_test, de_tokenizer, en_tokenizer, max_len)

if __name__ == '__main__':
    main()
