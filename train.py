import tensorflow as tf
import numpy as np
from model import create_multimodal_model
from data_preprocessing import preprocess_data
import os

def load_training_data():
    """
    Loads and preprocesses training data from manually added CSV/JSON files.
    For demonstration purposes, this function creates dummy data.
    
    In practice, replace the dummy data creation with real data extraction from the DataFrame.
    """
    # Example file paths – adjust to match your data folder and file names.
    file_paths = ['data/drug_data.csv', 'data/symptom_data.json']
    data = preprocess_data(file_paths)
    num_samples = len(data)
    
    # Dummy tokenized text: each sample is a sequence of 100 token IDs.
    text_data = np.random.randint(1, 5000, (num_samples, 100))
    # Dummy image data: random images of shape (224, 224, 3)
    image_data = np.random.random((num_samples, 224, 224, 3))
    # Dummy numerical data: e.g., 10 features per sample
    numerical_data = np.random.random((num_samples, 10))
    # Dummy labels: one-hot encoded for 3 classes
    labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, num_samples), num_classes=3)
    return (text_data, image_data, numerical_data, labels)

def train():
    # Load training data
    text_data, image_data, numerical_data, labels = load_training_data()

    # Create the multimodal model
    model = create_multimodal_model(vocab_size=5000, max_length=100, image_shape=(224,224,3), num_features=10)

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
