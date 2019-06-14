import requests
import cv2
from threading import Thread
from detector import DetectorAPI
import numpy as np
from collections import deque

class Camera(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.daemon = True
        self.running = True
        self.buffer = 32
        self.direction = ''
        self.pts = deque(maxlen=self.buffer)
        self.stream_url = 0 
        
        self.start()

    def personDetected(self, img):
        print("PERSON DETECTED")

    def stop(self):
        self.running = False

    def detectObject(self, img, counter):
        (dX, dY) = (0, 0)

        # loop over the set of tracked points
        for i in np.arange(1, len(self.pts)):
            # if either of the tracked points are None, ignore them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            # check to see if enough points have been accumulated in the buffer
            print("counter: {} - i: {}".format(counter, i))
            if counter >= 10 and i == 1 and self.pts[-10] is not None:
                print("IS NOT NONE")
                # compute the difference between the x and y
                # coordinates and re-initialize the direction text variables
                dX = self.pts[-10][0] - self.pts[i][0]
                dY = self.pts[-10][1] - self.pts[i][1]
                (dirX, dirY) = ("", "")

                # ensure there is significant movement in the x-direction
                if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX) == 1 else "West"
                # ensure there is significant movement in the y-direction
                if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"

                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    self.direction = "{}-{}".format(dirY, dirX)
                # otherwise, only one direction is non-empty
                else:
                    self.direction = dirX if dirX != "" else dirY

            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(self.buffer / float(i + 1)) * 2.5)
            cv2.line(img, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

        # show the movement deltas and the direction of movement on the frame
        cv2.putText(img, self.direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)
        cv2.putText(img, "dx: {}, dy: {}".format(dX, dY),
            (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)

    def run(self):
        model_path = '/home/bruno/Documentos/Bruno/Projetos/skill-monitoring/ssd_mobilenet_v1_coco/frozen_inference_graph.pb'
        odapi = DetectorAPI(path_to_ckpt=model_path)
        cap = cv2.VideoCapture(self.stream_url)
        threshold = 0.7
        counter = 0

        while self.running:
            r, img = cap.read()
            img = cv2.resize(img, (1280, 720))

            boxes, scores, classes, num = odapi.processFrame(img)

            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

                    center = (int((box[1]+box[3])/2), int((box[0]+box[2])/2))
                    cv2.circle(img, center, 5, (0, 0, 255), -1)
                    self.pts.appendleft(center)
                    self.personDetected(img)
            

            # OBJECT TRACKING
            self.detectObject(img, counter)
            
            counter += 1
            
            # Read image
            img = cv2.resize(img, (960, 540))
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                self.stop()
     
if __name__ == '__main__':
    c = Camera()
    c.join()