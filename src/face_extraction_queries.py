import sys
import numpy as np
import cv2
import json
import os
import glob
import tensorflow as tf

sys.path.append('../3rd_party/')

from ServiceMTCNN import detect_face

def detect_face_by_image(query, masks):
	# TF session
	sess = tf.Session()
	# Create MTCNN
	pnet, rnet, onet = detect_face.create_mtcnn(sess, None)	
	# MTCNN Hyperparameters
	minsize = 20
	threshold = [0.6, 0.7, 0.7]
	factor = 0.709

	ret = []
	
	for image, mask in zip(query, masks):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		im_height, im_width = image.shape[0], image.shape[1]
		
		# detected faces
		detected_faces, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
		best_bbox = None
		best_overlap = 0 
		for face_info in detected_faces:
			x, y, _x, _y = [int(coord) for coord in face_info[:4]] 
			bbox_mask = np.zeros((im_height, im_width), dtype=np.uint8)
			cv2.rectangle(bbox_mask, (x, y), (_x, _y), 255, cv2.FILLED)
			ones_mask = np.ones((im_height, im_width), dtype=np.uint8)
			res = cv2.bitwise_and(mask, bbox_mask, mask=ones_mask)	

			overlap = np.count_nonzero(res)

			if overlap > best_overlap:
				best_bbox = face_info
				best_overlap = overlap

		if best_bbox is not None:
			x, y, _x, _y = [int(coord) for coord in best_bbox[:4]]
			face = image[y : _y, x : _x]	
			face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
			cv2.imshow('fig', face)
			cv2.waitKey()
			cv2.destroyAllWindows()
			ret.append(face)

	return ret


def detect_face_by_path(query_path, masks_path):
	query = [cv2.imread(im_path) for im_path in query_path]
	masks = [cv2.imread(mask_path, 0) for mask_path in masks_path]
	return detect_face_by_image(query, masks)
	

if __name__ == "__main__":
	with open('../cfg/config.json', 'r') as f:
		config = json.load(f)

	queries_folder = os.path.abspath(config['raw_data']['queries_folder'])
	query_path = ['archie.1.src.bmp', 'archie.2.src.bmp', 'archie.3.src.bmp', 'archie.4.src.bmp']	
	query_path = [os.path.join(queries_folder, pth) for pth in query_path]
	masks_path = ['archie.1.mask.bmp', 'archie.2.mask.bmp', 'archie.3.mask.bmp', 'archie.4.mask.bmp']	
	masks_path = [os.path.join(queries_folder, pth) for pth in masks_path]

	detect_face_by_path(query_path, masks_path)
