import pandas as pd 
import numpy as np 
import tensorflow as tf
from keras import layers 
import matplotlib.pyplot as plt 
from IPython import display
from jiwer import wer 

data_path = "speech_model\dataset\LJSpeech-1.1"

wavs_path = "speech_model\dataset\LJSpeech-1.1\wavs"
metadata_path = "speech_model\dataset\LJSpeech-1.1\metadata.csv"

metadata_df = pd.read_csv(metadata_path,sep="|",header=None,quoting=3)

print(metadata_df.tail())
