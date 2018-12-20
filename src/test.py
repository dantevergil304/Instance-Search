from search import SearchEngine
from face_extraction_queries import detect_face_by_path
from default_feature_extraction import extract_feature_from_face
import os
import cv2
def test_remove_bad_face():
    name = "peggy"
    query_folder = "../data/raw_data/queries/"
    query_img = [
        name + ".1.src.bmp",
        name + ".2.src.bmp",
        name + ".3.src.bmp",
        name + ".4.src.bmp"
        
    ]
    query_mask = [
        name + ".1.mask.bmp",
        name + ".2.mask.bmp",
        name + ".3.mask.bmp",
        name + ".4.mask.bmp"
    ]

    search_eng = SearchEngine()
    query_img = [os.path.join(query_folder, img) for img in query_img]
    query_mask = [os.path.join(query_folder, mask) for mask in query_mask]
    faces = detect_face_by_path(query_img, query_mask)
    img_features = []
    for face in faces:
        feature = extract_feature_from_face(search_eng.default_vgg, face)
        img_features.append((face, feature))
    query = search_eng.remove_bad_faces(img_features)
    print("Total \"good\" faces : ", len(query))
    for q in query:
        cv2.imshow("q", q[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test_remove_bad_face()


