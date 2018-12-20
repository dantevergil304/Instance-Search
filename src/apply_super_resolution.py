import sys
import os
import cv2
import tensorflow as tf
import json

sys.path.append('../3rd_party/Image-Super-Resolution')
import models


def apply_super_res(true_img, model_type='rnsr', scale_factor=2, save_intermediate=False, mode='patch'):
	cwd = os.getcwd()
	# image_path = os.path.join(cwd, image_path)
	# print('Image path for applying Super-Resolution:', image_path)

	with open('../cfg/config.json', 'r') as f:
		config = json.load(f)
	im_super_res_folder = os.path.abspath(config['3rd_party']['Image_Super_Resolution'])
	
	os.chdir(im_super_res_folder)
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
	else:
            	model = models.DistilledResNetSR(scale_factor)
	
	image = model.upscale_direct(true_img, save_intermediate=save_intermediate, mode=mode, return_image=True)
	os.chdir(cwd)

	cv2.imshow('Super Resolution', image)
	cv2.waitKey()
	cv2.destroyAllWindows()
	return image

if __name__ == '__main__':
	true_img = cv2.imread('../data/processed_data/faces/queries/detect_before_mask/archie.2.face.bmp')
	apply_super_res(true_img)	
	pass
