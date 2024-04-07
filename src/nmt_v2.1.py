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
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tensorflow as tf
from keras.layers import Layer
import transformers

# Load transformer tokenizer and encoder 
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
transformer_encoder = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

# Bahdanau Attention Layer
class BahdanauAttention(Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units, kernel_regularizer=regularizers.l2(0.001))
    self.W2 = Dense(units, kernel_regularizer=regularizers.l2(0.001))
    self.V = Dense(1)

  def call(self, query, values):
    hidden_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

# Encoder 
encoder_inputs = Input(shape=(None,), dtype='int32')
enc_mask = tf.cast(tf.math.not_equal(encoder_inputs, 0), dtype='int32')
encoder_embeddings = transformer_encoder(encoder_inputs, attention_mask=enc_mask)[0] 

# Decoder
decoder_inputs = Input(shape=(None,), dtype='int32')
dec_mask = tf.cast(tf.math.not_equal(decoder_inputs, 0), dtype='int32')
decoder_embeddings = transformer_encoder(decoder_inputs, attention_mask=dec_mask)[0]

# Attention layer
attention = BahdanauAttention(64)
context_vector, attention_weights = attention(encoder_embeddings, decoder_embeddings)

# Concat attention context vector and decoder embeddings
decoder_concat_input = Concatenate(axis=-1)([context_vector, decoder_embeddings])

# Decoder GRU 
decoder_gru = GRU(256, return_sequences=True, return_state=True) 
decoder_outputs, _ = decoder_gru(decoder_concat_input)

# Dense layer
decoder_dense = Dense(5000, activation='softmax') 
decoder_outputs = decoder_dense(decoder_outputs)

# Define model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def load_data(filepath):
  with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
  return lines

def tokenize(lines, maxlen):
  tokenizer = Tokenizer(num_words=5000) 
  tokenizer.fit_on_texts(lines)
  sequences = tokenizer.texts_to_sequences(lines)
  padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post') 
  return padded_sequences, tokenizer

def add_start_end_tokens(lines):
  processed_lines = []
  for line in lines:
    processed_line = '<start> ' + line + ' <end>'
    processed_lines.append(processed_line)

  return processed_lines

def create_dataset(enc_lines, dec_lines):
  enc_padded, enc_tokenizer = tokenize(enc_lines, maxlen=20)
  dec_padded, dec_tokenizer = tokenize(dec_lines, maxlen=20)

  enc_vocab_size = len(enc_tokenizer.word_index) + 1
  dec_vocab_size = len(dec_tokenizer.word_index) + 1

  dataset = tf.data.Dataset.from_tensor_slices((enc_padded, dec_padded)).shuffle(2048).batch(128)

  return dataset, enc_vocab_size, dec_vocab_size, enc_tokenizer, dec_tokenizer

def train(dataset, enc_vocab_size, dec_vocab_size):
  
  checkpoint_filepath = 'model_checkpoint'
  model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
  train_data = dataset.take(8000) 
  val_data = dataset.skip(8000).take(2000)

  model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[model_checkpoint_callback])

  # Save model
  model.save('nmt_model')

def translate(input_text, enc_tokenizer, dec_tokenizer, model, maxlen):
  
  input_seq = enc_tokenizer.texts_to_sequences([input_text])
  input_seq = pad_sequences(input_seq, maxlen=maxlen)

  translated = ''
  target_seq = np.zeros((1,1)) 
  target_seq[0,0] = dec_tokenizer.word_index['<start>']

  while True:
    batch_predictions = model.predict([input_seq, target_seq])
    sampled_index = np.argmax(batch_predictions[0, -1, :])
    sampled_word = dec_tokenizer.index_word[sampled_index]
    if sampled_word == '<end>' or len(translated) > maxlen: 
      break
    translated += ' ' + sampled_word 
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = sampled_index
    target_seq = pad_sequences(target_seq, maxlen=maxlen)

  return translated

# Load data
enc_lines = load_data('data/train.envi.txt')
dec_lines = load_data('data/train.vi.txt')

enc_lines = add_start_end_tokens(enc_lines)
dec_lines = add_start_end_tokens(dec_lines)

dataset, enc_vocab, dec_vocab, enc_tokenizer, dec_tokenizer = create_dataset(enc_lines, dec_lines)

train(dataset, enc_vocab, dec_vocab)

# Test translation
input_text = 'This is a test'
translated = translate(input_text, enc_tokenizer, dec_tokenizer, model, 20)
print(translated)