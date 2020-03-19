import sys
import time
import os, os.path
import datetime
import numpy as np
import cv2
import argparse
from .face_detector.FaceDetector import FaceDetector

DIRNAME = os.path.dirname(os.path.abspath(__file__))
RECOGNIZED_PATH = '../../storage/recognized_faces/'
DETECTED_PATH = '../../storage/detected_faces/'
UNKNOWN_PATH = '../../storage/unknown_faces/'
SCREENSHOTS_PATH = '../../storage/screenshots/'
IS_DEV = os.getenv('PYTHON_ENV') == 'development'

class CameraManagement:
	def __init__(self, Recognizer, ImageCreator):
		self.detector = FaceDetector()
		#self.recognizer = Recognizer(self.detector)
		self.imageCreator = ImageCreator()
		self.window_name="Camera Management"
		#self.recognizer.setup()
		self.stream_url = 'rtsp://192.168.2.167:8554/profile1'

	def add_overlays(self, frame, faces, frame_rate):
		if faces is not None:
			for face in faces:
				bb = face['rect']
				cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
				if face['name'] is not None:
					cv2.putText(frame, face['name'], (bb[0], bb[3]),
					            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

		cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)


	def alert_unrecognized_faces(self, faces, frame):
		if faces:
			for face in faces:
				self.imageCreator.save_image(UNKNOWN_PATH, face['cropped'])
				print("ALERT_UNKNOWN_FACE: {} {}".format(face['distance'], face['id']))

	def alert_recognized_faces(self, recognized_faces, frame, frame_rate):
		if recognized_faces:
			for face in recognized_faces:
				self.imageCreator.save_image(RECOGNIZED_PATH, face['cropped'], face['name'])
				print("ALERT_RECOGNIZED_FACE: {} {} {}".format(face['name'], face['distance'], face['id']))

	def run(self):
		frame_interval = 30  # Number of frames after which to run face detection
		fps_display_interval = 5  # seconds
		frame_rate = 0
		frame_count = 0

		cap = cv2.VideoCapture(self.stream_url)
		#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
		#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
		#cap.set(cv2.CAP_PROP_FPS, 30)

		start_time = time.time()

		while True:
			detected_faces = None
			recognized_faces = None

			# Capture frame-by-frame
			ret, frame = cap.read()
			if (ret == 0):
				print("Error: check if webcam is connected.")
				return

			if ((frame_count % frame_interval) == 0):

				detected_faces = self.detector.detect_faces(frame)

				if detected_faces:
					if (IS_DEV):
						self.imageCreator.save_image(DETECTED_PATH, frame)

					recognized_faces = False #self.recognizer.recognize_faces(detected_faces)

					if recognized_faces:
						self.alert_recognized_faces(recognized_faces, frame, frame_rate)

						unrecognized_faces = [face for face in recognized_faces if face['recognized'] == False]

						if unrecognized_faces:
							self.alert_unrecognized_faces(unrecognized_faces, frame)

				# Check our current fps
				end_time = time.time()
				if (end_time - start_time) > fps_display_interval:
					frame_rate = int(frame_count / (end_time - start_time))
					start_time = time.time()
					frame_count = 0

			frame_count += 1

			if (IS_DEV):
				if (detected_faces is not None):
					# adiciona contorno na face detectada/reconhecida e nome caso seja reconhecida
					self.add_overlays(frame, recognized_faces or detected_faces, frame_rate)

				cv2.imshow(self.window_name, frame)

			keyPressed = cv2.waitKey(1) & 0xFF

			if keyPressed == 27: # ESC key
				break
			elif keyPressed == 13: # ENTER key
				self.imageCreator.save_image(SCREENSHOTS_PATH, frame)
				print('Screenshot saved!')
		# When everything is done, release the capture
		cap.release()
		cv2.destroyAllWindows()
