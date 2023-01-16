from datetime import datetime
import cv2,os,sys
import numpy as np
import pymongo
from os import listdir
from os.path import isfile, join


data_path = 'C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\public\\dataset\\'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# date = datetime.now().strftime("%d/%m/%Y")


Training_Data, Labels = [], []
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(int(os.path.split(files)[1].split(".")[0]))

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

# print("Model Training Complete!!!!!")


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["trackit"]
stud = db["students"]
ml = db["mlstatuses"]
cn = db["cnstatuses"]
iot = db["iotstatuses"]

face_classifier = cv2.CascadeClassifier('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\haarcascade.xml')

x=0
y=0
w=0
h=0
# def face_detector(img, size = 0.5):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray,1.3,5)

#     if faces is():
#         return img,[]

#     for(x,y,w,h) in faces:
#         cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
#         roi = img[y:y+h, x:x+w]
#         roi = cv2.resize(roi, (200,200))

#     return img,roi

cap = cv2.VideoCapture(0)

name = ""
course = ""
sem = ""
sid=0
flag = 0
roll = 0
date = datetime.now().strftime("%d/%m/%Y")
dict = {
    'Date': date,
    "140001": "Absent",
    "140002": "Absent",
    "140003": "Absent",
    "140004": "Absent",
    "140005": "Absent"
}
studData = []
req= sys.argv[1]
# req = "ml"
while True:

    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        face = []
    else:
        for(x,y,w,h) in faces:
            # cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = image[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            face = roi

    # image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        
        roll = result[0]

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
       
        if confidence > 80:
            if(dict[str(roll)] == "Absent"):
                    dict[str(roll)] = "Present"
            
            studData = stud.find({"id": roll},{"_id": 0})
            for item in studData:
                sid = item['id']
                name = item['name'] 
                course = item['course']
                sem = item['semester']

            cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,255),2)
            cv2.putText(image, str(sid) , (x+70, y+h+40), font, 0.5, (0,0,0), 2)
            cv2.putText(image,  name , (x+70, y+h+55), font, 0.5, (0,0,0), 2)
            cv2.putText(image,  course + " - " +sem , (x+60, y+h+70), font, 0.5, (0,0,0), 2)
            cv2.imshow('Face Cropper', image)           


        else:
            cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Taking Attendance', image)


    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Taking Attendance', image)
        pass

    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()

print(req)
print(name," ",course," ",sem)
if(req == "ml"):
    ml.insert_one(dict)
elif(req == "cn"):
    cn.insert_one(dict)
else:
    iot.insert_one(dict)