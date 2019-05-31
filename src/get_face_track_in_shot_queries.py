from deep_learning_utils import extendBB
import glob
import os
import subprocess
import cv2


if __name__ == '__main__':
    for query_dir in glob.glob('../data/raw_data/queries/2018/tv18.person.example.shots/*'):
        print(f'Processing query: {query_dir}')
        for query_shot in glob.glob(os.path.join(query_dir, '*mp4')):
            print(f'\t...Processing shot: {query_shot}')
            shots_json_path = os.path.splitext(query_shot)[0] + '.shots.json'
            track_txt_path = os.path.splitext(query_shot)[0] + '.track.txt'

            if not os.path.exists(shots_json_path):
                subprocess.call(
                    ['pyannote-structure.py', 'shot', '--verbose', query_shot, shots_json_path])
            if not os.path.exists(track_txt_path):
                subprocess.call(['pyannote-face.py', 'track', '--verbose',
                                 '--every=0.5', query_shot, shots_json_path, track_txt_path])

            face_track_dir = os.path.splitext(query_shot)[
                0] + '_facetrack'
            if not os.path.exists(face_track_dir):
                os.mkdir(face_track_dir)
            with open(track_txt_path, 'r') as f:
                for line in f:
                    t, identifier, left, top, right, bottom, status = line.rstrip().split()
                    t, left, top, right, bottom = float(t), float(
                        left), float(top), float(right), float(bottom)

                    identifier_dir = os.path.join(face_track_dir, identifier)
                    if not os.path.exists(identifier_dir):
                        os.mkdir(identifier_dir)

                    identifier_frame_dir = os.path.join(
                        identifier_dir, 'frames')
                    if not os.path.exists(identifier_frame_dir):
                        os.mkdir(identifier_frame_dir)
                    if not os.path.exists(os.path.join(identifier_frame_dir, f'{t}-{left}-{top}-{right}-{bottom}.png')):
                        if t < 10:
                            subprocess.call(
                                ['ffmpeg', '-i', query_shot, '-ss', f'00:00:0{t}', '-vframes', '1', os.path.join(identifier_frame_dir, f'{t}-{left}-{top}-{right}-{bottom}.png')])
                        elif t >= 10:
                            subprocess.call(
                                ['ffmpeg', '-i', query_shot, '-ss', f'00:00:{t}', '-vframes', '1', os.path.join(identifier_frame_dir, f'{t}-{left}-{top}-{right}-{bottom}.png')])

                    identifier_face_dir = os.path.join(identifier_dir, 'faces')
                    # if os.path.exists(identifier_face_dir):
                    #     subprocess.call(['rm', '-rf', identifier_face_dir])
                    if not os.path.exists(identifier_face_dir):
                        os.mkdir(identifier_face_dir)
                    save_path = os.path.join(identifier_face_dir,
                                             f'{t}-{left}-{top}-{right}-{bottom}.png')
                    if not os.path.exists(save_path):
                        frame = cv2.imread(os.path.join(
                            identifier_frame_dir, f'{t}-{left}-{top}-{right}-{bottom}.png'))
                        if frame is None:
                            continue
                        height, width, _ = frame.shape
                        _left = int(width * left)
                        _top = int(height * top)
                        _right = int(width * right)
                        _bottom = int(height * bottom)

                        _, _top, _, _bottom = extendBB(
                            (height, width), _left, _top, _right, _bottom)

                        face = frame[_top:_bottom, _left:_right]

                        cv2.imwrite(os.path.join(identifier_face_dir,
                                                 f'{t}-{left}-{top}-{right}-{bottom}.png'), face)
