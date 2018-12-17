from mtcnn.mtcnn import MTCNN

import cv2
import json
import os
import glob


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
		mask = cv2.imread(os.path.join(queries_folder, maskname), 0)

		res = cv2.bitwise_and(image, image, mask=mask)
		
		# cv2.imshow('res', res)
		# cv2.waitKey(0)
		# cv2.destryAllWindows()

		detected_faces = detector.detect_faces(res)
		if len(detected_faces) > 0:
			face_info = max(detected_faces, key=lambda x:x['confidence'])
			x, y, w, h = face_info['box']
			face = image[y : y + h, x : x + w]	
			cv2.imwrite(os.path.join(queries_face_folder, facename), face)
