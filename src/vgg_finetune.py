from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.models import load_model

import cv2
import json
import numpy as np
import tensorflow as tf
import os


def fine_tune(train_data, eps=1, model_name=None):
    '''
    Parameters:
    - train_data: [X, Y]
    - eps: num epochs
    Returns:
    - fine-tuned model
    '''
    with open('../cfg/config.json', 'r') as f:
        cfg = json.load(f)

    save_path = cfg['models']['VGG_folder']['VGG_fine_tuned_folder']
    nb_class = 2

    model = VGGFace(input_shape=(224, 224, 3))
    fc7_relu = model.get_layer('fc7/relu').output
    out = Dense(nb_class, activation='softmax', name='fc8')(fc7_relu)

    finetune_model = Model(model.input, out)

    finetune_model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    #cv2.imshow(str(train_data[1][0]), train_data[0][0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    train_images = np.reshape(train_data[0], (-1, 224, 224, 3))
    train_labels = np.array(train_data[1])

    finetune_model.fit(train_images, train_labels, epochs=eps, batch_size=10)
    finetune_model.save(os.path.join(save_path, model_name + '.h5'))

    return finetune_model


def extract_face_features(model_path, faces_list):
    model = load_model(model_path)

    feature_extractor = Model(model.input, model.get_layer('fc6').output)

    X = np.reshape(faces_list, (-1, 224, 224, 3))
    print(X.shape)

    return feature_extractor.predict(X, batch_size=20, verbose=1)
