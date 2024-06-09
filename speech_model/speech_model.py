import pandas as pd 
import numpy
import numpy as np 
import tensorflow as tf
from keras import layers 
import keras
import matplotlib.pyplot as plt 
from IPython import display
from jiwer import wer 

data_path = "E:\Project-Ava\speech_model\dataset\LJSpeech-1.1"

wavs_path = "E:\Project-Ava\speech_model\dataset\LJSpeech-1.1\wavs"
metadata_path = "E:\Project-Ava\speech_model\dataset\LJSpeech-1.1\metadata.csv"

metadata_df = pd.read_csv(metadata_path,sep="|",header=None,quoting=3)

metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df =metadata_df[["file_name", "normalized_transcription"]]
metadata_df =metadata_df.sample(frac=1).reset_index(drop=True)

split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'!?"]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

frame_length = 256 
frame_step = 160 
fft_length = 334 

def encode_single_sample(wav_file,label):
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio,axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectogram = tf.signal.stft(
        audio,frame_length=frame_length,frame_step=frame_step, fft_length=fft_length
    )
    spectogram = tf.abs(spectogram)
    spectogram = tf.math.pow(spectogram, 0.5)
    
    means =tf.math.reduce_mean(spectogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectogram, 1, keepdims=True)
    spectogram = (spectogram - means)/ (stddevs + 1e-10)
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)
    
    return spectogram,label 


# training dataset 
batch_size = 32 
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["file_name"]) , list(df_train["normalized_transcription"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]) , list(df_val["normalized_transcription"]))
)

validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

def CTCLoss(y_true,y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64") 
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64") 
    
    loss = tf.keras.backend.ctc_batch_cost(y_true,y_pred, input_length, label_length)
    return loss 

def build_model(input_dim, output_dim, rnn_layers = 5, rnn_units = 128):
    input_spectogram = layers.Input((None,input_dim), name= "input")
    x = layers.Reshape((-1, input_dim, 1), name ="expand_dim")(input_spectogram)
    x = layers.Conv2D(
        filters = 32,
        kernel_size=[11,21],
        strides=[2,2],
        padding="same",
        use_bias=False,
        name = "conv_1",
    )(x)
    x= layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    x =layers.Conv2D(
        filters = 32,
        kernel_size=[11,21],
        strides=[1,2],
        padding="same",
        use_bias=False,
        name = "conv_2",
    )(x)
    x= layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x =layers.Bidirectional(
            recurrent, name = f"bidirectional_{i}", merge_mode= "concat"
        )(x)
        if i < rnn_layers:
            x =layers.Dropout(rate=0.5)(x)
    x =layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x =layers.ReLU(name="dense_1_relu")(x)
    x =layers.Dropout(rate=0.5)(x)
    
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    model = keras.Model(input_spectogram, output, name="speech-deep")
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(optimizer=opt, loss=CTCLoss)
    return model 

model = build_model(
    input_dim=fft_length// 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=512,
)
model.summary(line_length=110)

epochs = 100

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
)
