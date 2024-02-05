import cv2
import sys
from random import randint

tracker_types= ['BOOSTING','MIL','KCF','TLD','MEDISNFLOW','MOSSE','CSRT']

def create_tracker(tracker):
    if tracker==tracker_types[0]:
        track=cv2.legacy.TrackerBoosting_create()
    elif tracker==tracker_types[1]:
        track=cv2.legacy.TrackerMIL_create()
    if tracker==tracker_types[2]:
        track=cv2.legacy.TrackerKCF_create()
    if tracker==tracker_types[3]:
        track=cv2.legacy.TrackerTLD_create()
    if tracker==tracker_types[4]:
        track=cv2.legacy.TrackerMedianFlow_create()
    if tracker==tracker_types[5]:
        track=cv2.legacy.TrackerMOSSE_create()
    if tracker==tracker_types[6]:
        track=cv2.legacy.TrackerCSRT_create()
    else:
        track=None
        print('Unvalid')

    return track

# print(create_tracker('CSRT'))

# video=cv2.VideoCapture('pexels-pavel-danilyuk-5790075 (1080p).mp4')
video=cv2.VideoCapture('VIDEO1.1.mov')
if not video.isOpened():
    print('Error in loading video')
    sys.exit()

ok,frame=video.read()

bboxes=[]
colors=[]

while True:
    bbox=cv2.selectROI('MultiTracker',frame)
    bboxes.append(bbox)
    colors.append((randint(0,255),randint(0,255),randint(0,255)))
    k=cv2.waitKey(0) & 0XFF
    if k==113: #Q - quit
        break

print(bboxes)
print(colors)

tracker_type='CSRT'
multi_tracker= cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    multi_tracker.add(create_tracker(tracker_type),frame,bbox)

while video.isOpened():
    ok,frame=video.read()
    if not ok:
        break

    ok,boxes=multi_tracker.update(frame)

    for i, new_box in enumerate(boxes):
        (x,y,w,h)=[int(v) for v in new_box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),colors[i],2)

    cv2.imshow('MultiTracker',frame)
    if cv2.waitKey(1) & 0XFF==27: # esc
        break