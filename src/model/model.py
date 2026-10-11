import keras.layers as layers
from keras import Input, Model
import tensorflow as tf

# ====================================================================================
# ====================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! # of Parameter: 4145 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ====================================================================================
# ====================================================================================


# def get_model(shape, num_classes):
#     inputs = Input(shape=shape)

#     x = layers.Conv2D(12, (2, 2), activation="tanh")(inputs)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(8, (2, 2), activation="tanh")(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.DepthwiseConv2D((2, 2), activation="tanh")(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(8, (2, 2), activation="relu")(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(12, (2, 2), activation="tanh", padding="same")(x)
#     x = layers.MaxPooling2D((2, 2), padding="same")(x)
#     x = layers.Flatten()(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Dense(8, activation="relu")(x)

#     actv_fnc = "sigmoid" if num_classes == 1 else "softmax"
#     outputs = layers.Dense(num_classes, activation=actv_fnc)(x)

#     model = Model(inputs, outputs)
#     return model

# ====================================================================================
# ====================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! # of Parameter: 5453 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ====================================================================================
# ====================================================================================


def get_model(shape, num_classes):
    inputs = Input(shape=shape)
    
    x = layers.Conv2D(16, (2, 2), activation="tanh")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (2, 2), activation="tanh")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.DepthwiseConv2D((2, 2), activation="tanh")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (2, 2), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (2, 2), activation="tanh", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    
    # Reshape for GRU: Treat the width dimension as timesteps, height*channels as features
    # Input shape after conv: (batch, height, width, channels)
    # Reshape to (batch, width, height*channels) for GRU (timesteps=width)
    x = layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[1] * tf.shape(x)[3])))(x)
    
    x = layers.GRU(32, return_sequences=False)(x)  # Lightweight GRU, no return sequences for classification
    
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(8, activation="relu")(x)
    
    actv_fnc = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(num_classes, activation=actv_fnc)(x)
    
    model = Model(inputs, outputs)
    return model