import cv2
import sys

def getDuration(input_path):  
    # duration = total_frame / FPS
    cap = cv2.VideoCapture(input_path)
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

if __name__ == '__main__':
    input_path = sys.argv[1]
    print(getDuration(input_path))

