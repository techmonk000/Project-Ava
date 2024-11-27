import numpy as np 
import tensorflow as tf
from keras import layers, activations, models, preprocessing, utils
import keras
import os
import yaml
import pickle
from gensim.models import Word2Vec
import re

dir_path = './dataset'
questions, answers = [], []

files_list = os.listdir(dir_path)

for filepath in files_list:
    if filepath.endswith('.yml'):
        stream = open(dir_path + os.sep + filepath, 'r', encoding='utf-8')
        docs = yaml.safe_load(stream)
        conversations = docs.get('conversations', [])

        for con in conversations:
            if len(con) > 2:
                questions.append(con[0])
                answers.append(' '.join(con[1:]))
            elif len(con) > 1:
                questions.append(con[0])
                answers.append(con[1])

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1
print(f'VOCAB SIZE : {VOCAB_SIZE}')

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

tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max(len(x) for x in tokenized_questions)
padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
encoder_input_data = np.array(padded_questions)
print(encoder_input_data.shape, maxlen_questions)

tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max(len(x) for x in tokenized_answers)
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_input_data = np.array(padded_answers)
print(decoder_input_data.shape, maxlen_answers)

for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
decoder_output_data = np.array(onehot_answers)
print(decoder_output_data.shape)

encoder_inputs = layers.Input(shape=(maxlen_questions,))
encoder_embedding = layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = layers.LSTM(200, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = layers.Input(shape=(maxlen_answers,))
decoder_embedding = layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
decoder_lstm = layers.LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(VOCAB_SIZE, activation=activations.softmax)
output = decoder_dense(decoder_outputs)

model = models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=50, epochs=150)
model.save('model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved.")
