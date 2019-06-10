from sklearn.svm import LinearSVC
from feature_extraction import extract_feature_from_face
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import GlobalAveragePooling2D
from keras import backend as K

import pickle
import json
import os
import time


def isGoodFace_classifier(face_img, classifier_type="linear_svm_fc7"):
    with open('../cfg/config.json', 'r') as f:
        config = json.load(f)
    with open(os.path.join(
            config["models"]["Good_bad_face_classifier_folder"], f'{classifier_type}.pkl'), 'rb') as f:
        classifier = pickle.load(f)

    extract_feat_t = time.time()
    # Get Deep Model for extracting features
    if classifier_type == 'linear_svm_fc7':
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        out = vgg_model.get_layer('fc7').output
        model = Model(vgg_model.input, out)
    elif classifier_type == 'linear_svm_fc6':
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        out = vgg_model.get_layer('fc6').output
        model = Model(vgg_model.input, out)
    elif classifier_type == 'linear_svm_pool5_gap':
        model = VGGFace(input_shape=(224, 224, 3),
                        pooling='avg', include_top=False)
    elif classifier_type == 'linear_svm_pool4_gap':
        vgg_model = VGGFace(input_shape=(224, 224, 3), pooling='avg')
        pool4 = vgg_model.get_layer('pool4').output
        out = GlobalAveragePooling2D()(pool4)
        model = Model(vgg_model.input, out)
    elif classifier_type == 'linear_svm_resnet50':
        vgg_model = VGGFace(input_shape=(224, 224, 3),
                            pooling='avg', model='resnet50')
        out = vgg_model.get_layer('flatten_1').output
        model = Model(vgg_model.input, out)

    feat = extract_feature_from_face(model, face_img)
    K.clear_session()
    extract_feat_t = time.time() - extract_feat_t

    score = classifier.decision_function(feat)
    print('Clf score:', score)
    if classifier.predict(feat) == 0:
        return True, extract_feat_t
    return False, extract_feat_t
