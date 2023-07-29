import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import librosa as lr

# Enable mixed precision training
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

# Load the data
df = pd.read_csv('validated.tsv', sep="\t")

# Remove special characters
def RMChars(y):
    chars = ['!', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':',
             ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    return ''.join(ch for ch in y if ch not in chars)

# Preprocess the text data
outs = df["sentence"].astype(str).apply(RMChars)

# Tokenization and sequence padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(outs)
outs = tokenizer.texts_to_sequences(outs)
outs = pad_sequences(outs)

# Load the audio data
audio_list = []
for path in df["path"]:
    x, sr = lr.load(f"clips/{path}".replace('mp3', 'wav'), sr=160)
    audio_list.append(x)

X = np.array(pad_sequences(audio_list))
y = outs

# Expand dimensions
X = np.expand_dims(X, -1)

# Save preprocessed data
np.save("X.npy", X)
np.save("y.npy", y)

def create_transformer_model(input_shape, output_dim, num_heads, ff_dim, num_blocks):
    inputs = keras.Input(shape=input_shape)

    # Transformer blocks
    x = inputs
    for _ in range(num_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[1]
        )(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward neural network
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(input_shape[1])(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # LSTM layer
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Output layer
    outputs = layers.Dense(output_dim, dtype='float32')(x)  # Ensure the final dtype is float32

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Model configuration
input_shape = (2110, 1)
output_dim = np.max(y) + 1  # Adjust output dimension based on the maximum label value
num_heads = 2
ff_dim = 16
num_blocks = 3

# Create the model
model = create_transformer_model(input_shape, output_dim, num_heads, ff_dim, num_blocks)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# Create the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle and batch the dataset
batch_size = 32  # Adjust batch size based on available memory
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Train the model
model.fit(train_dataset, epochs=10)
