# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Function to read a single .mat file
def read_mat_file(file_path):
    try:
        mat_data = scipy.io.loadmat(file_path)
        eeg_data = mat_data['paddedData']
        return eeg_data
    except KeyError:
        print(f"Key 'paddedData' not found in {file_path}")
        return None
    except OSError as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to read labels from xlsx
def read_labels(xlsx_path):
    try:
        labels_df = pd.read_excel(xlsx_path)
        labels = {row['Name']: row['classes'] for _, row in labels_df.iterrows()}
        return labels
    except Exception as e:
        print(f"Error reading {xlsx_path}: {e}")
        return {}

# Generator function to load data in batches
def data_generator(folder_path, labels, batch_size, num_classes):
    file_names = sorted(os.listdir(folder_path))
    total_files = len(file_names)
    while True:
        for start in range(0, total_files, batch_size):
            end = min(start + batch_size, total_files)
            batch_data = []
            batch_labels = []
            for file_name in file_names[start:end]:
                if file_name.endswith('.mat'):
                    file_path = os.path.join(folder_path, file_name)
                    eeg_data = read_mat_file(file_path)
                    if eeg_data is not None and file_name in labels:
                        label = labels[file_name]
                        batch_data.append(eeg_data)
                        batch_labels.append(label)
            if batch_data and batch_labels:
                batch_data = np.array(batch_data)
                batch_labels = np.array(batch_labels)
                batch_labels = to_categorical(batch_labels, num_classes)
                batch_data = batch_data[..., np.newaxis]
                yield batch_data, batch_labels

# Function to handle sessions
def handle_session(folder_path, xlsx_path, batch_size, num_classes):
    labels = read_labels(xlsx_path)
    sample_file = os.path.join(folder_path, sorted(os.listdir(folder_path))[0])
    sample_data = read_mat_file(sample_file)
    if sample_data is not None:
        input_shape = sample_data.shape + (1,)
        print("Determined input shape:", input_shape)
    else:
        print("Unable to determine input shape. Please check the data.")
        raise ValueError("Unable to determine input shape.")
    data_gen = data_generator(folder_path, labels, batch_size, num_classes)
    num_files = len(os.listdir(folder_path))
    steps_per_epoch = num_files // batch_size
    return data_gen, input_shape, steps_per_epoch

session_paths = {
    'Session1': (r'C:\Users\PC\Documents\Research Work\Data\Output_segF\Output_segF\Sessions\Session1\Equilize', r'C:\Users\PC\Documents\Research Work\Data\Output_segF\Output_segF\Sessions\Session1\Classes_session_1.xlsx'),
    'Session2': (r'C:\Users\PC\Documents\Research Work\Data\Output_segF\Output_segF\Sessions\Session2\Equilizer', r'C:\Users\PC\Documents\Research Work\Data\Output_segF\Output_segF\Sessions\Session2\Classes_session_2.xlsx'),
    'Session3': (r'C:\Users\PC\Documents\Research Work\Data\Output_segF\Output_segF\Sessions\Session3\Equilizer', r'C:\Users\PC\Documents\Research Work\Data\Output_segF\Output_segF\Sessions\Session3\Classes_session_3.xlsx')
}

batch_size = 32  # Adjust batch size as needed
num_classes = 4  # Set the number of classes

# Process each session
for session_name, (folder_path, xlsx_path) in session_paths.items():
    print(f"Processing {session_name}...")
    data_gen, input_shape, steps_per_epoch = handle_session(folder_path, xlsx_path, batch_size, num_classes)
    
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(data_gen, steps_per_epoch=steps_per_epoch, epochs=10)

    plt.plot(history.history['accuracy'])
    plt.title(f'Model accuracy for {session_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title(f'Model loss for {session_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    print(f"Completed processing {session_name}")
