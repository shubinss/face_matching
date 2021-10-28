from flask import Flask, request, jsonify
import model_app

# initialize models detectors
face_detector_box = model_app.MTCNN()
face_features_extractor = model_app.VGGFace(include_top = False, model = 'resnet50', pooling='avg')
threshold_match = 0.6 
# create db know face
meta_face_db = model_app.create_meta_db(face_detector_box, face_features_extractor)     
# initialize Flask application
app = Flask(__name__)
# API that returns JSON
@app.route('/', methods=['POST'])
def get_detections():
    source = request.form.get('source')
    idx = request.form.get('idx')
    face_for_matching_origin = model_app.open_cam(source, face_detector_box)
    json_for_matching, face_for_matching, features_for_matching = model_app.face_features(face_detector_box, face_features_extractor, face_for_matching_origin)
    error_msg, error_identification, score, json_for_matching, certan_face_db = model_app.inference(meta_face_db, json_for_matching, face_for_matching, features_for_matching, idx)
    inference_result = model_app.inference_json(error_msg, error_identification, score, json_for_matching, certan_face_db, face_for_matching_origin, threshold_match)
    return jsonify(inference_result.to_json(orient = 'columns'))

if __name__ == '__main__':
    app.run(debug = False, host = '0.0.0.0', port = 5000)