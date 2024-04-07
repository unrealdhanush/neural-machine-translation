import os
import time
import argparse
import numpy as np
import keras
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from bahdanau_attention import BahdanauAttention
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from preprocess import preprocess, load_dataset, preprocess_sentence


def max_length(tensor):
    '''
        From TensorFlow NMT tutorial
    '''
    return max(len(t) for t in tensor)



def loss_function(real, pred):
    '''
        From TensorFlow NMT tutorial
    '''
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = keras.metrics.sparse_categorical_crossentropy(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def evaluate(sentence, inp_lang, targ_lang, max_length_targ, max_length_inp, encoder, decoder):
    '''
        From TensorFlow NMT tutorial
    '''
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, 1024))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + ' '
        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def translate(sentence, inp_lang, targ_lang, max_length_targ, max_length_inp, encoder, decoder):
    '''
        From TensorFlow NMT tutorial
    '''
    result, sentence, _ = evaluate(sentence, inp_lang, targ_lang, max_length_targ, max_length_inp, encoder, decoder)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    return sentence


def nmt(input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, 
        inp_lang, targ_lang, fun, translate_sentence, max_length_target, max_length_input,
        data_path, num_examples, ckpt):
    import keras
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        
        example_input_batch, example_target_batch = next(iter(dataset))
    
        # Encoder
        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
        sample_hidden = encoder.initialize_hidden_state()
        sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
        print (f'Encoder output shape: (batch size, sequence length, units): {sample_output.shape}')
        print (f'Encoder Hidden state shape: (batch size, units): {sample_hidden.shape}')

        # Attention layer
        attention_layer = BahdanauAttention(10)
        attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
        print(f'Attention result shape: (batch size, units): {attention_result.shape}')
        print(f'Attention weights shape: (batch_size, sequence_length, 1): {attention_weights.shape}')    

        # Decoder
        decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
        print (f'Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}')
    
        optimizer = keras.optimizers.legacy.Adam()
        
        checkpoint_dir = ckpt
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        
        if args.fun == 'train': 
            print('Training..')
            EPOCHS = 10
            for epoch in range(EPOCHS):
                start = time.time()
                enc_hidden = encoder.initialize_hidden_state()
                total_loss = 0
                for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                    batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, optimizer, BATCH_SIZE)
                    total_loss += batch_loss

                # saving (checkpoint) the model every 2 epochs
                if (epoch + 1) % 2 == 0:
                    checkpoint.save(file_prefix = '/Users/dhanush/Northeastern/2023 fall/EECE7398/HW_2/ckpt')

                print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                  total_loss / steps_per_epoch))
        print(f'Model saved in file {checkpoint_prefix}')
    
        if args.fun == 'test':
            print('Testing..')
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

            en = []
            vi = []
            with open(os.path.join('./data/', 'tst2012.en'), 'r')  as f:
                for line in f:
                    en.append(line)
            with open(os.path.join('./data/', 'tst2012.vi'), 'r')  as f:
                for line in f:
                    vi.append(line)         
            #en = en[:100]
            #vi = vi[:100]
            bleu_scores = []
            
            for idx, line in enumerate(en):
                result, sentence, attention_plot = evaluate(line, inp_lang, targ_lang, max_length_targ, max_length_inp, encoder, decoder)
                sm = SmoothingFunction()
                score = sentence_bleu([sentence], vi[idx], smoothing_function=sm.method1)*2
                bleu_scores.append(score)
                if idx % 10 == 0:
                    print(f'{line}\t{sentence}\n\t{result}')
                    print(f'{idx} bleu score: {score}\n')
            avg_score = np.average(bleu_scores)
            print(f'Average BLEU score: {avg_score}')

        if args.fun == 'translate':
            print('Restoring model..')
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print('Translating..')
            translate(translate_sentence,inp_lang, targ_lang, max_length_targ, max_length_inp, encoder, decoder) 
    

def train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, optimizer, BATCH_SIZE):
    '''
        From TensorFlow NMT tutorial
    '''
    import tensorflow as tf
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/')
    parser.add_argument('--num_examples', type=int, default=30000)
    parser.add_argument('--fun', type=str, default="train")
    parser.add_argument('--translate', type=str, default='hello world!')
    parser.add_argument('--id_gpu', type=int, default=-1)
    parser.add_argument('--ckpt', type=str, default='/ckpt')
    args = parser.parse_args()

    if args.id_gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.id_gpu)
    
    print('Processing and loading data...')
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, inp_lang, target_lang = preprocess(args.data_path, args.num_examples)
    max_length_targ, max_length_inp = max_length(target_tensor_train), max_length(input_tensor_train)   
 
    with open('./data/translate.txt', 'w') as f:
        f.write(args.translate) 
    tensor, lang_tokenizer = load_dataset('./data/translate.txt')

    nmt(input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, 
        inp_lang, target_lang, args.fun, args.translate, max_length_targ, max_length_inp, 
        args.data_path, args.num_examples, args.ckpt)
