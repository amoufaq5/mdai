import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model

def create_text_branch(vocab_size=5000, max_length=100):
    # Input for text data (e.g., tokenized patient responses)
    text_input = Input(shape=(max_length,), name="text_input")
    x = Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length)(text_input)
    x = LSTM(64)(x)
    x = Dense(32, activation='relu')(x)
    return text_input, x

def create_image_branch(input_shape=(224,224,3)):
    # Input for image data (e.g., CT/MRI scans)
    image_input = Input(shape=input_shape, name="image_input")
    x = Conv2D(32, (3,3), activation='relu')(image_input)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return image_input, x

def create_numerical_branch(input_dim):
    # Input for numerical data (e.g., lab test results, vital signs)
    num_input = Input(shape=(input_dim,), name="num_input")
    x = Dense(32, activation='relu')(num_input)
    x = Dense(16, activation='relu')(x)
    return num_input, x

def create_multimodal_model(vocab_size=5000, max_length=100, image_shape=(224,224,3), num_features=10):
    """
    Builds a multimodal TensorFlow model that fuses text, image, and numerical inputs.
    The output is a 3-class softmax representing:
      - OTC recommendation
      - Refer to doctor
      - Further evaluation required
    """
    text_input, text_branch = create_text_branch(vocab_size, max_length)
    image_input, image_branch = create_image_branch(image_shape)
    num_input, num_branch = create_numerical_branch(num_features)

    # Concatenate all feature branches
    combined = concatenate([text_branch, image_branch, num_branch])
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax', name='diagnosis_output')(x)

    model = Model(inputs=[text_input, image_input, num_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
