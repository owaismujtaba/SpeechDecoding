import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Model, Input
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import GRU, Input, Conv1D, MaxPooling1D, concatenate, Dense, Flatten, Reshape

import pdb

import config as config

early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=5,    # Number of epochs to wait before stopping
    restore_best_weights=True,  # Restore best weights when stopping
    verbose=1
)


class NeuroInceptDecoder:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        

    def inception_module(self, input_tensor, filters):
        conv_1x1 = Conv1D(filters, 1, padding='same', activation='relu')(input_tensor)
        conv_3x3 = Conv1D(filters, 3, padding='same', activation='relu')(input_tensor)
        conv_5x5 = Conv1D(filters, 5, padding='same', activation='relu')(input_tensor)

        max_pool = MaxPooling1D(3, strides=1, padding='same')(input_tensor)
        max_pool = Conv1D(filters, 1, padding='same', activation='relu')(max_pool)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)
        
        return output

    def create_model(self):
        input_layer = Input(shape=(self.input_shape[0], 1))

        # Inception Module 1
        x = self.inception_module(input_layer, 64)

        # GRU Module
        x = GRU(128, return_sequences=True)(x)
        x = GRU(256, return_sequences=True)(x)
        x = GRU(512, return_sequences=False)(x)

        x = Reshape((1, 512))(x)

        # Inception Module 2
        x = self.inception_module(x, 128)

        x = Flatten()(x)

        # Fully Connected Layers
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)

        output_layer = Dense(self.output_shape, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, y_train):
        self.model = self.create_model()
        
        self.model.fit(X_train, y_train,
            batch_size=2048, 
            epochs=100, 
            validation_split=0.10,
            callbacks=[early_stopping]
        )





class NeuroInceptDecoder1:
    def __init__(self, input_shape=(2247, ), output_shape=23, gru_units=64):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.gru_units = gru_units


    def _create_model(self):

        inputs = Input(shape=(self.input_shape[0], 1))   # Add channel dimension (F, 1)

        # **First Inception Module**
        x = self.inception_module(inputs)

        # **GRU Module**
        x = layers.GRU(self.gru_units, return_sequences=True)(x)
        x = layers.GRU(self.gru_units * 2, return_sequences=True)(x)
        x = layers.GRU(self.gru_units * 4, return_sequences=False)(x)

        # **Second Inception Module**
        x = layers.Reshape((x.shape[-1], 1))(x)  # Reshape to apply Conv1D
        x = self.inception_module(x)

        # **Flatten before Fully Connected Layers**
        x = layers.Flatten()(x)

        # **Dense Layers**
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)

        # **Output Layer**
        outputs = layers.Dense(self.output_shape, activation='linear')(x)

        # **Create Model**
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse')

    def inception_module(self, input_tensor):
        
        """Applies Inception-style convolutions to the input tensor."""
        conv_1x1 = layers.Conv1D(64, 1, padding='same', activation='relu')(input_tensor)
        conv_3x3 = layers.Conv1D(64, 3, padding='same', activation='relu')(input_tensor)
        conv_5x5 = layers.Conv1D(64, 5, padding='same', activation='relu')(input_tensor)

        max_pool = layers.MaxPooling1D(3, strides=1, padding='same')(input_tensor)
        max_pool_conv = layers.Conv1D(64, 1, padding='same', activation='relu')(max_pool)

        return layers.Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, max_pool_conv])

    def train(self, X_train, y_train):
        self._create_model()
        X_train = X_train.reshape(X_train.shape[0], self.input_shape[0], 1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=config.BATCH_SIZE, 
            epochs=config.EPOCHS, 
            validation_split=0.10,
            callbacks=[early_stopping]
        )

