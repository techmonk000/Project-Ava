import numpy as np
import tensorflow as tf
from keras import models, preprocessing
import keras
import pickle
import re 
model = models.load_model('model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

VOCAB_SIZE = len(tokenizer.word_index) + 1
maxlen_questions = model.input[0].shape[1]
maxlen_answers = model.input[1].shape[1]

vocab = [word for word in tokenizer.word_index]

def tokenize(sentences):
    tokens_list, vocabulary = [], []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)
    return tokens_list, vocabulary

def make_inference_models():
    encoder_inputs = model.input[0]
    encoder_states = model.get_layer(index=2).output[1:]

    encoder_model = models.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_state_input_h = keras.layers.Input(shape=(200,))
    decoder_state_input_c = keras.layers.Input(shape=(200,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding = model.get_layer(index=3)(decoder_inputs)
    decoder_lstm = model.get_layer(index=4)
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = model.get_layer(index=5)
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

enc_model, dec_model = make_inference_models()

def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = [tokenizer.word_index.get(word, 0) for word in words]
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

while True:
    user_input = input("Enter question: ")
    if user_input.lower() == 'quit':
        break

    states_values = enc_model.predict(str_to_tokens(user_input))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index.get('start', 0)

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None

        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += f' {word}'
                sampled_word = word
                break

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation.strip())
