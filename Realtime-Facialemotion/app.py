from flask import Flask, Response, render_template
import cv2
from keras.saving.save import load_model
from keras_preprocessing.image.utils import img_to_array
import numpy as np
from flask import Flask, request, session, redirect, url_for, render_template






app=Flask(__name__)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)

def generate_frames():  
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            label = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction = classifier.predict(roi)[0]
                    label=emotion_labels[prediction.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    print(label)
                   
                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
             
            
            _,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


@app.route('/')
def demo():
    return render_template('demo.html')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)
   