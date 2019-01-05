from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

import numpy as np
import tensorflow as tf


def fine_tune(train_data, eps=5):
    '''
    Parameters:
    - train_data: [X, Y]
    - eps: num epochs
    Returns:
    - fine-tuned model
    '''
    nb_class = 2

    model = VGGFace(input_shape=(224, 224, 3))
    fc7_relu = model.get_layer('fc7/relu').output
    out = Dense(nb_class, activation='softmax', name='fc8')(fc7_relu)

    finetune_model = Model(model.input, out)

    finetune_model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    train_images = np.reshape(train_data[0], (-1, 224, 224, 3))
    train_labels = np.array(train_data[1])

    finetune_model.fit(train_images, train_labels, epochs=eps)

    return finetune_model
