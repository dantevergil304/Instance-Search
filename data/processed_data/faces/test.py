import cv2
import pickle
import sys
with open(sys.argv[1], "rb") as f:
	faces = pickle.load(f)

cv2.imshow("face", faces[int(sys.argv[2])-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
