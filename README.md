# Neural Machine Translation with Bahdanau Attention Mechanism

This project implements a neural machine translation (NMT) system using the Bahdanau attention mechanism. The NMT system is capable of translating text from English to Vietnamese and English to Dutch, as well as vice versa. The system consists of an encoder-decoder architecture with attention mechanism, trained on the Stanford machine translation datasets.

## Overview

Neural machine translation (NMT) is an approach to machine translation that uses neural networks to predict the likelihood of a sequence of words in the target language, given a sequence of words in the source language. The Bahdanau attention mechanism allows the model to focus on different parts of the input sequence when generating each word in the output sequence.

## Components

### Encoder

The encoder component of the NMT system processes the input sequence (source language) and generates a fixed-length vector representation, which captures the semantics of the input sequence. In this project, we use a bidirectional LSTM (Long Short-Term Memory) network as the encoder.

### Decoder

The decoder component of the NMT system takes the encoder's output vector and generates the output sequence (target language) word by word. At each step, the decoder utilizes the attention mechanism to focus on different parts of the input sequence. We use an LSTM network as the decoder.

### Bahdanau Attention Mechanism

The Bahdanau attention mechanism allows the decoder to selectively focus on different parts of the input sequence during the decoding process. This attention mechanism improves the model's ability to translate long sentences accurately.

## Usage

To use the trained NMT model for translation:

1. Clone or download the repository.
2. Install the necessary dependencies (e.g., TensorFlow, NumPy).
3. Ensure you have the required dataset files in the 'data' directory.
4. Run the following command to train the model:
```python nmt.py train```

This will train the model using the provided training data.

5. Once the model is trained, you can test it by running the following command:
```python nmt.py test```

This will evaluate the model's performance on the test data and display the translated sentences along with their BLEU scores.

## License

This project is licensed under the [MIT License](LICENSE).
