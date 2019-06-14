import requests
import cv2
from threading import Thread
from .detector import DetectorAPI
import requests

class Camera(Thread):
    def __init__(self):
        print("INITIALIZING CAMERA...")
        Thread.__init__(self)
        self.daemon = True
        self.running = True
        self.start()
    

        self.stream_url = 'rtsp://192.168.2.167:8554/profile0' 

    def personDetected(self, img):
        print("PERSON DETECTED")


    def stop(self):
        self.running = False

    def run(self):
        model_path = '/home/bruno/Documentos/Projetos/octopus/skill_monitoring/ssd_mobilenet_v1_coco/frozen_inference_graph.pb'
        odapi = DetectorAPI(path_to_ckpt=model_path)
        cap = cv2.VideoCapture(self.stream_url)
        threshold = 0.7

        while self.running:
            print("READING")
            r, img = cap.read()
            img = cv2.resize(img, (1280, 720))

            boxes, scores, classes, num = odapi.processFrame(img)

            # Visualization of the results of a detection.

            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                    self.personDetected(img)
                              # Read image
            img = cv2.resize(img, (960, 540))
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                self.stop()
     