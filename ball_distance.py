from roboflow import Roboflow
import cv2 as cv
import math


rf = Roboflow(api_key="zK2fRpek9w6HHU4wymYj")
project = rf.workspace().project("konstruktionswettbewerb")
model = project.version(2).model

f = cv.VideoCapture(0)

while(f.isOpened()):
  # f.read() methods returns a tuple, first element is a bool 
  # and the second is frame
    ret, frame = f.read()
    if ret == True:
        # save frame as a “temporary” jpeg file
        cv.imwrite('temp.jpg', frame)
        # run inference on “temporary” jpeg file (the frame)
        predictions = model.predict('temp.jpg')
        predictions_json = predictions.json()
        # printing all detection results from the image
        # print(predictions_json)
        if (len(predictions_json['predictions'])>0):
          width = predictions_json['predictions'][0]['width']
          height = predictions_json['predictions'][0]['height']
          x= width * height
          center = (predictions_json['predictions'][0]['x'], predictions_json['predictions'][0]['y'])
          print(1795*math.pow(x, -0.492))
          cv.rectangle(frame, (int(center[0]-width/2), int(center[1]-height/2)), (int(center[0]+width/2), int(center[1]+height/2)), (0,0,255), 2)
          cv.imshow('detect', frame)
          cv.waitKey(1)
          