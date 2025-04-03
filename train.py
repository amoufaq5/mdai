import os
import logging
import tensorflow as tf
import numpy as np
from model import create_multimodal_model
from data_preprocessing import preprocess_data

def load_training_data():
    """
    Loads and preprocesses all CSV, JSON, Excel (.xls/.xlsx) files in the 'data/' folder.
    Returns tokenized text data, image data, numerical data, and labels (dummy for now).
    """
    data_dir = 'data'
    # Gather all files in 'data/' folder that match CSV, JSON, XLS, XLSX
    file_paths = []
    for file_name in os.listdir(data_dir):
        if file_name.lower().endswith(('.csv', '.json', '.xls', '.xlsx')):
            file_paths.append(os.path.join(data_dir, file_name))

    # Preprocess to get unified DataFrame (could be empty if no valid files exist or they're empty)
    df = preprocess_data(file_paths)
    num_samples = len(df)

    if num_samples == 0:
        logging.warning("No data found or data folder is empty. Using dummy data for demonstration.")
        num_samples = 10
        text_data = np.random.randint(1, 5000, (num_samples, 100))
        image_data = np.random.random((num_samples, 224, 224, 3))
        numerical_data = np.random.random((num_samples, 10))
        labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, num_samples), num_classes=3)
    else:
        logging.info(f"Loaded {num_samples} samples from data folder.")
        
        # -------------------------------------------------------------------
        # You can replace the dummy array creation below with real logic
        # to convert your DataFrame `df` into numeric arrays for training.
        # For example, if `df` has a column 'text_input', you’d tokenize it.
        # If you have columns for images, you'd load them from paths, etc.
        # -------------------------------------------------------------------

        # Dummy arrays for demonstration:
        text_data = np.random.randint(1, 5000, (num_samples, 100))
        image_data = np.random.random((num_samples, 224, 224, 3))
        numerical_data = np.random.random((num_samples, 10))
        labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, num_samples), num_classes=3)

    return text_data, image_data, numerical_data, labels

def train():
    # Load training data from 'data/' folder
    text_data, image_data, numerical_data, labels = load_training_data()

    # Create the multimodal model
    model = create_multimodal_model(
        vocab_size=5000, 
        max_length=100, 
        image_shape=(224,224,3), 
        num_features=10
    )

    # Train the model
    model.fit(
        [text_data, image_data, numerical_data],
        labels,
        epochs=10,
        batch_size=16,
        validation_split=0.2
    )

    # Save the trained model
    os.makedirs('saved_model', exist_ok=True)
    model.save('saved_model/diagnosis_model')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train()
