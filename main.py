import cv2
import torch
from tracker import *
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#video
cap = cv2.VideoCapture(r'files\petal_20240407_200250.mp4')

#camera
#cap = cv2.VideoCapture(0)


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

tracker = Tracker()

#get id of person inside the area
'''area_1 = [(377,315),(429,373),(535,339),(500,296)]
area1 = set()'''

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (300, 700))
    #cv2.polylines(frame, [np.array(area_1, np.int32)], True, (255,0,0),3)
    results = model(frame)
#    frame = np.squeeze(results.render())
    list = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

#find person in class
        b = str(row['name'])
        if 'person' in b:
            list.append([x1, y1, x2, y2])

    boxes_ids = tracker.update(list)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.rectangle(frame, (x,y),(w,h),(255,255,0),2)
        text = 'person id: ' + str(id)
        cv2.putText(frame, 'person', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        #result = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(w),int(h)), False)

        '''if result>0:
            area1.add(id)
    print(area1)'''

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()