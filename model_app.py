#%% library
import pathlib
import pandas as pd
from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
import os
import json
import datetime
import time
#%% face features extraction function
def face_features(detector_box, detector_features, image):
    json_description = detector_box.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if json_description != []:
        x1, y1, width, heigh = json_description[0]['box']
        x2, y2 = x1 + width, y1 + heigh
        face_array = [np.expand_dims(np.asarray(Image.fromarray(image[y1:y2, x1:x2]).resize((244, 244))).astype('float32'), axis = 0)]
        features = [detector_features.predict(preprocess_input(face_array[0], version = 2))]
    else:
        json_description, face_array, features = '', '', ''
    return json_description, face_array, features
#%% match face function
def is_match(face_db, candidate_features, error_identification, error_msg, threshold_match = 0.5):
    score = cosine(face_db.iloc[0]['features'][0], candidate_features[0])
    if score <= threshold_match:
        error_identification = False 
    else:
        error_identification = True 
        error_msg.append('Не установлено соотвествие с БД')
    return score, error_identification, error_msg
#%% inference function
def inference(meta_face_db, json_for_matching, face_for_matching, features_for_matching, idx):
    error_msg = []
    error_identification = True
    score = []
    meta_face = []
    if len(json_for_matching)==0:
        error_msg.append('нет лиц в кадре')
    if len(json_for_matching)>1:
        error_msg.append('больше одного лица в кадре')
    if meta_face_db[meta_face_db['ID']==idx].empty:
        error_msg.append('идентификационный номер отсутствует в БД')  
    else:
        if meta_face_db[meta_face_db['ID']==idx]['features'].tolist()[0] == '':
            error_msg.append('изображение в БД повреждено')   
        if len(error_msg)==0:
            error_identification = False
            score, error_identification, error_msg = is_match(meta_face_db[meta_face_db['ID']==idx], features_for_matching, error_identification, error_msg, threshold_match = 0.5)
            meta_face = meta_face_db[meta_face_db['ID']==idx].iloc[0]
    if score==[]:
        score = 1
    return error_msg, error_identification, score, json_for_matching, meta_face
#%% json inference function
def inference_json(error_msg, error_identification, score, json_for_matching, certan_face_db, face_for_matching, threshold_match):
    if len(certan_face_db)!=0:
        name, position, ID = certan_face_db['name'], certan_face_db['position'], certan_face_db['ID']
    else:
        name, position, ID = None, None, None
    inference = pd.DataFrame(data = [[error_identification, error_msg, 1-score, len(json_for_matching), name, position, ID, threshold_match]], columns=['identification error', 'cause of error', 'confidence of the system in the similarity', 'number of faces in the frame', 'employee name', 'position', 'position id', 'accuracy threshold'])
    folder = {True: 'failure', False: 'success'}[error_identification]
    name_json = f'.\\result\\{folder}\\inference_{ID}_{str(datetime.datetime.now().strftime("%H_%M_%S"))}.json'
    name_capture = f'.\\result\\{folder}\\inference_{ID}_{str(datetime.datetime.now().strftime("%H_%M_%S"))}.jpeg'
    with open(os.path.normpath(name_json), 'w') as outfile:
        json.dump(inference.to_json(orient = 'columns'), outfile)
    capture_save = Image.fromarray(face_for_matching.astype(np.uint8))
    capture_save.save(name_capture)
    return inference
#%% meta db function
def create_meta_db(face_detector_box, face_features_extractor):
    face_db_list = list(pathlib.Path().resolve().glob('db/*.jpg'))
    meta_face_db = pd.DataFrame([[file.stem, file.name, file, '', '', '', '', '', ''] 
                                 for file in face_db_list], 
                                columns = ['idx', 'file_name', 'path', 'json', 
                                           'face', 'features', 'ID', 'name', 'position'])
    for idx, image_path in enumerate(meta_face_db['path']):
        meta_face_db.loc[idx, ['json', 'face', 'features']] = face_features(face_detector_box, face_features_extractor, plt.imread(image_path))
        tmp_name = meta_face_db.loc[idx, 'idx'].split('_', 3)
        meta_face_db.loc[idx, ['ID', 'name', 'position']] = tmp_name[0], tmp_name[1], tmp_name[2]
    return meta_face_db
#%% camera capture functions 
def open_cam(source, detector_box):
    while True:
        if source.isdigit():
            if cv2.VideoCapture(int(source)).read()[0]==True:
                tic = time.perf_counter()
                frame = capture_cam(cv2.VideoCapture(int(source)), detector_box, tic)
                print('Image captured.')
                break
        else:
            if os.path.exists(os.path.normpath(source)):
                frame = plt.imread(os.path.normpath(source))
                print('Image received.')
                break
            else:
                print(f'Warning: unable to open source: {source}.')
    return frame

def capture_cam(capture, detector_box, tic):
    while capture.isOpened():
        ret, frame = capture.read()
        frame_added = frame.copy()
        json_description = detector_box.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if json_description != []:
            keypoints = json_description[0]['keypoints']
            frame_added = cv2.circle(frame_added, (keypoints['left_eye']), 2, (0,155,255), 2)
            frame_added = cv2.circle(frame_added, (keypoints['right_eye']), 2, (0,155,255), 2)
            frame_added = cv2.circle(frame_added, (keypoints['nose']), 2, (0,155,255), 2)
            frame_added = cv2.circle(frame_added, (keypoints['mouth_left']), 2, (0,155,255), 2)
            frame_added = cv2.circle(frame_added, (keypoints['mouth_right']), 2, (0,155,255), 2)
            x1, y1, width, heigh = json_description[0]['box']
            x2, y2 = x1 + width, y1 + heigh
            frame_added = cv2.rectangle(frame_added, (x1, y1) , (x2, y2), (13,214,53), 2)
        time_delay = int(time.perf_counter()-tic)
        frame_added = cv2.imshow('Video Window', cv2.putText(frame_added, f'Press "Q" or wait {10 - time_delay} sec to capture', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2))
        if (cv2.waitKey(20) & 0xFF == ord('q')) or (time_delay >= 10):
            capture.release()
            cv2.destroyAllWindows()
            break
    return frame
#%% initialize function
def main(source, idx):
    # initialize models detectors
    face_detector_box = MTCNN()
    face_features_extractor = VGGFace(include_top = False, model = 'resnet50', pooling='avg')
    # create db know face
    meta_face_db = create_meta_db(face_detector_box, face_features_extractor)
    # inference
    threshold_match = 0.6
    face_for_matching_origin = open_cam(source, face_detector_box)
    json_for_matching, face_for_matching, features_for_matching = face_features(face_detector_box, face_features_extractor, face_for_matching_origin)
    error_msg, error_identification, score, json_for_matching, certan_face_db = inference(meta_face_db, json_for_matching, face_for_matching, features_for_matching, idx)
    inference_result = inference_json(error_msg, error_identification, score, json_for_matching, certan_face_db, face_for_matching_origin, threshold_match)
    return inference_result, error_msg, error_identification, score, json_for_matching, certan_face_db, face_for_matching

