import cv2
import os
import datetime

DIRNAME = os.path.dirname(os.path.abspath(__file__))

class ImageCreator:

	def get_time(self):
		return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

	def save_image(self, path, frame, lastName=None):
		cv2.imwrite(os.path.join(DIRNAME, path) + self.get_time() +
		            ("_" + lastName if lastName != None else "") + ".jpg", frame)
