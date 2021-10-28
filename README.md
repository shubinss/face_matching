# Demo face matching
## Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt
```
## Add new employees to the database 
Add new images to folder "db", names of new files - ID_FullName_Position.jpeg.
## Running the Flask App (http://localhost:5000/)
Initialize and run the Flask app on port 5000 of your local machine by running the following command from the root directory of this repo.
```bash
python app.py
```
You should see the following appear in the command prompt if the app is successfully running.
```bash
 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://192.168.1.106:5000/ (Press CTRL+C to quit)
```
 This endpoint (http://localhost:5000/) takes in images or web cam as input and returns a JSON and save JSON and image result in ones of next folders - "./result/success" or "./result/failure".
For test APIs using Postman.
## Accessing Detections API with Postman 
Access the http://localhost:5000/ API through Postman by doing the following.
metod -> POST
check "Body"
check "form-data"
KEY -> source Value-> 0 or 1 (webcam) or file_name.jpeg (image)
KEY -> idx Value-> target ID

![response](https://github.com/shubinss/face_matching/blob/4f127743f652bcf5899116c1c2b330d85e7373db/postman_access.jpeg)

