# import numpy as np
# import cv2
# import face_recognition
# import os
#
# def allgivenimages(images):
#     ls=[];
#     for item in images:
#         currpath='imagesgiven/'+item;
#         img=face_recognition.load_image_file(currpath);
#         ls.append(face_recognition.face_encodings(img)[0]);
#     return ls;
#
# images=os.listdir("D:\gargt\Attendance Mark Project\src\imagesgiven");
# encodings=allgivenimages(images);
#
# imgTushar=face_recognition.load_image_file('imagesgiven/Shaurya.jpeg')
# imgTushar=cv2.cvtColor(imgTushar,cv2.COLOR_BGR2RGB)
# faceloc=face_recognition.face_locations(imgTushar)[0];
# encodeTushar=face_recognition.face_encodings(imgTushar)[0];
# cv2.rectangle(imgTushar,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),0)
#
# imgTusharTest=face_recognition.load_image_file('imagetotest/TusharNew.jpg')
# imgTusharTest=cv2.cvtColor(imgTusharTest,cv2.COLOR_BGR2RGB)
# facelocTest=face_recognition.face_locations(imgTusharTest)[0];
# encodeTusharTest=face_recognition.face_encodings(imgTusharTest)[0];
# cv2.rectangle(imgTusharTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),0)
#
# # results=face_recognition.compare_faces(encodings,encodeTusharTest);
# # print(results);
# video_capture = cv2.VideoCapture(0)
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#     img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     facecurrTest=face_recognition.face_locations(img);
#     encodecurrTest=face_recognition.face_encodings(img,facecurrTest);
#     # cv2.imshow('Unknown',frame);
#     for index,item in enumerate(encodings):
#         if(face_recognition.compare_faces(item,encodecurrTest)==True):
#             print(images[index]);
#     cv2.imshow('Unknown',frame);
# # cv2.imshow('Tushar',imgTushar)
# # cv2.imshow('TusharTest',imgTusharTest)
# cv2.waitKey(0)
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# from PIL import ImageGrab

path = 'imagesgiven'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
#img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matches);
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
#print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)