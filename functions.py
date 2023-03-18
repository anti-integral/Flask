import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import torch
print(cv2.__version__)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#for custom Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt')
classes=[]
f=open('classes.txt','r')
for line in f:
	classes.append(line.strip())

#for custom Model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
def yolo(img_path):
  img=cv2.imread(img_path)
  img=cv2.resize(img, (416,416))
  results = model(img)
  print(results.pandas().xyxy[0])
  Label,Bbox,Confidence=[],[],[]
  for res in results.pandas().xyxy:
      # print(len(res))
      for obj in range(len(res)):
          if res['confidence'][obj]>0.2:
              (x1, y1, x2, y2) = (res['xmin'][obj],res['ymin'][obj],res['xmax'][obj],res['ymax'][obj])
              bbox=(x1,y1,x2,y2)
              className = classes[res['class'][obj]]
              Label.append(className)
              Bbox.append(bbox)
              Confidence.append(res['confidence'][obj])
            #   cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (250, 19, 10),2)
  return Label,Bbox,Confidence
