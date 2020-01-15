from vision.skills.core import VisionSkill
import cv2
import time
import numpy
import glob
import logging
from detector import DetectorAPI

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.DEBUG)
logger = logging.getLogger('skill-monitoring')

class MonitoringSkill(VisionSkill):
	def __init__(self):
		super(MonitoringSkill, self).__init__(name="MonitoringSkill")

		self.threshold = 0.7
        self.lock = True

    def personDetected(self, img):
        logger.debug("DETECTED!")

    def stopSkill(self):
        self.lock = False


	def handleError(self, error):
		logging.error("Exception occurred", exc_info=True)

	def initialize(self, message):
        params = message.data
	    model_path = 'ssd_mobilenet_v1_coco/frozen_inference_graph.pb'
        odapi = DetectorAPI(path_to_ckpt=model_path)
        cap = cv2.VideoCapture(params['cameraId'])

        while self.Lock:
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

            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break



def create_skill():
	return MonitoringSkill()