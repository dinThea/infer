from source.yolo import YOLO
from source.Tracker import MultiTracker
from source.firebase import Firebase
import cv2

def capture_and_save(url):

    try:

        stream = cv2.VideoCapture(url)
        net = YOLO()
        kt = MultiTracker()
        fire = Firebase()

        while stream.isOpened():
            
            flag, frame = stream.read()
            bbox_net, classes, scores = net.detect(frame)
            center_list = []
            bbox_list = []
            class_list = []

            for bbox, class_in, score in izip(bbox_net, classes, scores):

                bbox_list.append(bbox)
                vis_frame = cv2.rectangle(vis_frame,
                                            (int(bbox[0]), int(bbox[1])),
                                            (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                                            (255,0,0),
                                            4)            
                center = kt.get_bbox_center(bbox)
                center_list.append(center)
                class_list.append(class_in)

            fire.create_data({'bbs': bbox_list, 'labels': class_list})

    except:
        print ('could not connect to stream')
    finally:
        stream.release()