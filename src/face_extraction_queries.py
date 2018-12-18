from mtcnn.mtcnn import MTCNN

import numpy as np
import cv2
import json
import os
import glob

METHOD = 'detect_before_mask'

if __name__ == "__main__":
	with open('../cfg/config.json', 'r') as f:
		config = json.load(f)

	queries_folder = os.path.abspath(config["raw_data"]["queries_folder"])
	queries_face_folder = os.path.abspath(config["processed_data"]["queries_face_folder"])

	detector = MTCNN()

	for query in glob.glob(os.path.join(queries_folder, '*.src.bmp')):
		imagename = query.split('/')[-1]
		maskname = '.'.join([imagename.split('.')[0], imagename.split('.')[1],
				'mask', 'bmp'])
		facename = '.'.join([imagename.split('.')[0], imagename.split('.')[1],
				'face', 'bmp'])

		image = cv2.imread(query)
		im_height, im_width = image.shape[0], image.shape[1]
		mask = cv2.imread(os.path.join(queries_folder, maskname), 0)

		if METHOD == 'mask_before_detect':
			res = cv2.bitwise_and(image, image, mask=mask)
		
			detected_faces = detector.detect_faces(res)
			if len(detected_faces) > 0:
				face_info = max(detected_faces, key=lambda x:x['confidence'])
				x, y, w, h = face_info['box']
				face = image[y : y + h, x : x + w]	
				cv2.imwrite(os.path.join(queries_face_folder, METHOD, facename), face)
		elif METHOD == 'detect_before_mask':
			detected_faces = detector.detect_faces(image)
			best_bbox = None
			best_overlap = 0 
			for face_info in detected_faces:
				x, y, w, h = face_info['box']
				bbox_mask = np.zeros((im_height, im_width), dtype=np.uint8)
				cv2.rectangle(bbox_mask, (x, y), (x+w, y+h), 255, cv2.FILLED)
				ones_mask = np.ones((im_height, im_width), dtype=np.uint8)
				res = cv2.bitwise_and(mask, bbox_mask, mask=ones_mask)	

				overlap = np.count_nonzero(res)

				if overlap > best_overlap:
					best_bbox = face_info['box']
					best_overlap = overlap
				#cv2.imshow('fig', res)
				#cv2.waitKey()
				#cv2.destroyAllWindows()

			if best_bbox is not None:
				x, y, w, h = best_bbox 
				face = image[y : y + h, x : x + w]	
				cv2.imwrite(os.path.join(queries_face_folder, METHOD, facename), face)
