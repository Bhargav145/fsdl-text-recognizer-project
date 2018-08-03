from typing import Tuple
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D,Reshape
from tensorflow.keras import layers


def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int=128,
        dropout_amount: float=0.3,
        num_layers: int=3) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]
    #print("input_shape",input_shape)
    input_shape = np.array(input_shape)
    #input_shape = np.expand_dims(input_shape, axis=0)

    model = Sequential()
    # Don't forget to pass input_shape to the first layer of the model
    ##### Your code below (Lab 1)
    model.add(Reshape((28, 28,1), input_shape=(28,28)))
    model.add(Conv2D(32,(3,3),activation='relu',input_shape = input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))    
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dropout(dropout_amount))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes,activation='softmax'))
    ##### Your code above (Lab 1)

    return model

