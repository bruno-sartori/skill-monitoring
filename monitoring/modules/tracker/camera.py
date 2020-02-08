# -*- coding: utf-8 -*-

import requests
import cv2
from threading import Thread
from .detector import DetectorAPI
import numpy as np
from collections import deque
from .movement import Movement
import time
import sys
import os

env = os.environ['PYTHON_ENV']

class Camera():
    def __init__(self, move=False, direction=False):
        self.move = move
        self.direction = direction
        self.running = True
        self.buffer = 32
        self.pts = deque(maxlen=self.buffer)
        self.area = [250, 150, 800, 600]
        self.stream_url = 0 #'rtsp://192.168.2.167:8554/profile1'
        self.isMoving = False
        self.isInside =  False
        self.lastNotificationSend = 0
        self.detectedTime = 0
        self.noDetectedTime = 0
        self.detectThreshold = 10 # seconds
        self.noDetectThreshold = 3

        if (self.move):
            self.movement = Movement()

    def sendAlert(self):
        url = os.environ['CORE_HOST'] + '/alerts/3' # level 3 alert
        try:
            response = requests.post(url,
                data={ 'origin': 'CAMERA', 'location': 5, 'title': 'ALERTA!!!', 'description': 'Teste' },
                headers={ 'Authorization': 'JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MSwibG9naW4iOiJicnVub3NhcnRvcmkud2VibWFzdGVyQGdtYWlsLmNvbSIsImNsaWVudCI6MSwiZGF0ZSI6IjIwMTktMDgtMTZUMDE6MzA6MTMuMDM0WiJ9.9z14DPcEAGUEE1MF27-BuBsRZHUav0jbNBy-zhYwVlU' },
                verify=False)

            print(response)
        except requests.exceptions.RequestException as error:
            print(error)

    def checkDetectedTime(self):
        now = time.time()

        # a person stayed in camera's area for more time than the threshold
        if (self.detectedTime != 0 and now - self.detectedTime > self.detectThreshold):
            if (self.lastNotificationSend == 0):
                self.lastNotificationSend = time.time()
                self.sendAlert()
            elif (time.time() - self.lastNotificationSend > self.detectThreshold):
                self.lastNotificationSend = time.time()
                self.sendAlert()

    def noPersonDetected(self):
        now = time.time()

        if (self.noDetectedTime != 0):
            #print("NO detected time: {}".format(now - self.noDetectedTime))

            if (now - self.noDetectedTime > self.noDetectThreshold):
                self.detectedTime = 0 # zera se não encontrar nenhuma pessoa por +3s
        elif (self.noDetectedTime == 0):
            self.noDetectedTime = time.time()


    # TODO: handle multiple detected persons
    def personDetected(self, img):
        if (self.detectedTime == 0):
            self.detectedTime = time.time()
        cv2.imwrite('./detected/' + str(time.time()) + '.jpg', img)


        self.noDetectedTime = 0
        #print("Detected time: {}".format(time.time() - self.detectedTime))

    def stop(self):
        self.running = False

    def rectContains(self, rect, pt):
        logic = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
        return logic

    def trackPerson(self, img, counter):
        (dX, dY) = (0, 0)
        direction = ''

        # loop over the set of tracked points
        for i in np.arange(1, len(self.pts)):
            # if either of the tracked points are None, ignore them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            try:
                # check to see if enough points have been accumulated in the buffer
                if counter >= 10 and i == 1 and self.pts[-10] is not None:
                    # compute the difference between the x and y
                    # coordinates and re-initialize the direction text variables
                    dX = self.pts[-10][0] - self.pts[i][0]
                    dY = self.pts[-10][1] - self.pts[i][1]
                    (dirX, dirY) = ("", "")

                    # ensure there is significant movement in the x-direction
                    if np.abs(dX) > 20:
                        if np.sign(dX) == 1:
                            dirX = 'RIGHT'
                        else:
                            dirX = 'LEFT'
                    # ensure there is significant movement in the y-direction
                    if np.abs(dY) > 20:
                        if np.sign(dY) == 1:
                            dirY = 'UP'
                        else:
                            dirY = 'DOWN'

                    # handle when both directions are non-empty
                    if dirX != "" and dirY != "":
                        direction = "{}-{}".format(dirY, dirX)

                    # otherwise, only one direction is non-empty
                    else:
                        direction = dirX if dirX != "" else dirY
            except IndexError as e:
                print("Error: %s" % e)
                pass
            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(self.buffer / float(i + 1)) * 2.5)
            cv2.line(img, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)


        if (self.move):
            if (self.isInside):
                print("CONTAINS")
                if (self.isMoving):
                    print("STOP")
                    self.isMoving = False
                    self.movement.move('STOP')
            else:
                print("MOVING {}".format(direction))
                self.isMoving = True
                self.movement.move(direction)
                self.movement.move('STOP')

        # show the movement deltas and the direction of movement on the frame
        cv2.putText(img, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)
        cv2.putText(img, "dx: {}, dy: {}".format(dX, dY),
            (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)

    def run(self):
        model_path = './ssd_mobilenet_v1_coco/frozen_inference_graph.pb' if env == 'production' else './monitoring/modules/tracker/ssd_mobilenet_v1_coco/frozen_inference_graph.pb'
        odapi = DetectorAPI(path_to_ckpt=model_path)
        cap = cv2.VideoCapture(self.stream_url)
        threshold = 0.7
        counter = 0


        while self.running:
            initTime = time.time()
            r, img = cap.read()

            if r == True:
                img = cv2.resize(img, (1280, 720))

                if (self.direction):
                    cv2.rectangle(img, (self.area[0],self.area[1]),(self.area[2], self.area[3]),(0,255,0),3)

                boxes, scores, classes, num = odapi.processFrame(img)

                possiblyDetected = len(boxes)

                # Visualization of the results of a detection.
                for i in range(len(boxes)):
                    # Class 1 represents human
                    if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

                        if (self.direction):
                            center = (int((box[1]+box[3])/2), int((box[0]+box[2])/2))
                            cv2.circle(img, center, 5, (0, 0, 255), -1)
                            self.isInside = self.rectContains(self.area, center)
                            self.pts.appendleft(center)

                        self.personDetected(img)
                    else:
                        possiblyDetected -= 1

                if (possiblyDetected == 0):
                    self.noPersonDetected()
                else:
                    self.checkDetectedTime()
                    if self.direction:
                        # TODO: handle for multiple person detected
                        self.trackPerson(img, counter)
                        counter += 1

                print("Elapsed time: {}".format(time.time() - initTime))
                # Read image
                if (env == 'development'):
                    img = cv2.resize(img, (960, 540))
                    cv2.imshow("preview", img)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q'):
                    self.stop()
            else:
                break
