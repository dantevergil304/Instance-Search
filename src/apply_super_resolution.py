import sys
import os
import cv2
import tensorflow as tf
import keras
import json

sys.path.append('../3rd_party/Image-Super-Resolution')
import models


def apply_super_res(true_img, model_type='sr', scale_factor=2, save_intermediate=False, mode='patch'):
    '''
    Parameters:
    - true_img: input image that you want to apply super-resolution
    Returns:
    - image: the resulting image after apply super-resolution
    '''
    cwd = os.getcwd()
    # image_path = os.path.join(cwd, image_path)
    # print('Image path for applying Super-Resolution:', image_path)

    with open('../cfg/config.json', 'r') as f:
        config = json.load(f)
    im_super_res_folder = os.path.abspath(
        config['3rd_party']['Image_Super_Resolution'])

    os.chdir(im_super_res_folder)
    keras.backend.set_image_data_format('channels_first')
    if model_type == "sr":
        model = models.ImageSuperResolutionModel(scale_factor)
    elif model_type == "esr":
        model = models.ExpantionSuperResolution(scale_factor)
    elif model_type == "dsr":
        model = models.DenoisingAutoEncoderSR(scale_factor)
    elif model_type == "ddsr":
        model = models.DeepDenoiseSR(scale_factor)
    elif model_type == "rnsr":
        model = models.ResNetSR(scale_factor)
    elif model_type == "distilled_rnsr":
        model = models.DistilledResNetSR(scale_factor)

    image = model.upscale_direct(
        true_img, save_intermediate=save_intermediate, mode=mode, return_image=True)
    keras.backend.set_image_data_format('channels_last')
    os.chdir(cwd)

    # cv2.imshow('Super Resolution', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return image


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    true_img = cv2.imread(
        '../data/raw_data/queries/chelsea.1.src.png')
    apply_super_res(true_img)
    pass
