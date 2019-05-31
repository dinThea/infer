import time

from pydarknet import Detector, Image
import cv2

class YOLO:
    # Optional statement to configure preferred GPU. Available only in GPU version.
    # pydarknet.set_cuda_device(0)

    def __init__(self):
        self.net = Detector(bytes("cfg/danyolo.cfg", encoding="utf-8"), bytes("weights/danyolo_900.weights", encoding="utf-8"), 0,
                       bytes("cfg/obj.names", encoding="utf-8"))

            # Only measure the time taken by YOLO and API Call overhead
    def detect(self, frame):
        
        dark_frame = Image(frame)
        results = self.net.detect(dark_frame)
        del dark_frame
        bboxes = []
        cats = []
        scores = []
        for cat, score, bounds in results:
            x, y, w, h = bounds
            bboxes.append((int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)))
            cats.append(cat)
            scores.append(score)

        return bboxes, cats, scores