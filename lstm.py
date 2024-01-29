import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


embedding_dim = 100
max_sequence_length = 256 # Maximum sequence length
vocab_size = 100000  # Maximum number of words to keep
tokenizer_file = 'tokenizer.pkl'
model_file = "lstm_model.h5"
# train_data_file = 'all_train.csv'
train_data_file = 'new_train.csv'

def prepare_data(datasize):
    # Data Preparation
    data = pd.read_csv(train_data_file)
    data = data[0:datasize]
    # Step 5: Split the data into training and testing setsdata['text']
    texts = data['text']
    labels = data['label']
    return texts, labels

def train_tokenizer(texts):
    # train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # Tokenization and Sequencing
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    save_tokenizer(tokenizer)


def save_tokenizer(tokenizer):
    with open(tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer():
    # Load the tokenizer from the file
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
        return tokenizer

def train(texts, labels):
    model = Sequential()
    tokenizer = load_tokenizer()
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5, verbose=1, mode='max')
    history = model.fit(padded_sequences, labels, epochs=24, validation_split=0.2, callbacks=[early_stopping])
    print(history.history.keys())
    # Save the trained model
    model.save(model_file)
    show_history(history)

def predict(new_texts):
    # Inference
    model = load_model(model_file)
    tokenizer = load_tokenizer()
    sequences = tokenizer.texts_to_sequences(new_texts)
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])
    predictions = model.predict(new_padded_sequences)
    threshold = 0.5  # Threshold value for classification
    binary_predictions = np.where(predictions > threshold, 1, 0)
    return binary_predictions 

def do_train():
    TRAIN_DATE_SIZE = -1
    texts, labels = prepare_data(TRAIN_DATE_SIZE)
    train_tokenizer(texts)
    train(texts, labels)


def show_history(history):
    # Plot training history
    pprint(history.history.keys())
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot as an image file
    plt.tight_layout()
    plt.savefig('training_chart.png')

if __name__ == "__main__":
    do_train()