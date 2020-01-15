import requests
import cv2
from threading import Thread
from detector import DetectorAPI
import numpy as np
from collections import deque
from movement import Movement


def run():
    cap = cv2.VideoCapture('rtsp://192.168.2.167:8554/profile1')
    
    while True:
        r, img = cap.read()
        cv2.imshow("PREVIEW", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    run()    
