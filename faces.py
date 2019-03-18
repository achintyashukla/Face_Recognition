import numpy as np
import cv2
import pickle

face_cascde = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

lables = {}
with open("labels.pickle", "rb") as f:
    og_lables = pickle.load(f)
    lables = {v:k for k,v in og_lables.items()}

cap = cv2.VideoCapture(0)

while(True):
    #Capture frame by frame
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascde.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x,y,w,h) in faces:
        #print (x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #region of interest
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45: #and conf <= 85:
            #print(id_)
            #print(lables[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = lables[id_]
            color = (255, 255, 255)
            strok = 2
            cv2.putText(frame, name, (x,y), font, 1, color, strok, cv2.LINE_AA)

        img_item = "11.png"
        
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) #BGR 0-255
        strok = 2

        end_cord_x = x + w
        end_cord_y = y + h

        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, strok)

    #Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()