import sys
import time
import os, os.path
import datetime
import tensorflow as tf
import numpy as np
from . import facenet
import cv2
import argparse
from monitoring.align import detect_face
from .recognizer import Recognizer

DIRNAME = os.path.dirname(os.path.abspath(__file__))

class CameraManagement:
	def __init__(self):
		# some constants kept as default from facenet
		self.minsize = 20
		self.threshold = [0.6, 0.7, 0.7]
		self.factor = 0.709
		self.margin = 44
		self.input_image_size = 160
		self.window_name="teste"

	def alert_unrecognized_faces(self, faces, frame):
		print("ALERT: unrecognized faces found!")
		if faces:
			for face in faces:
				print("saving face.")
				cropped_face = face['cropped']
				img_name = os.path.join(DIRNAME, "../storage/unknown_faces/", datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
				cv2.imwrite(img_name, cropped_face)

				print("ALERT_UNKNOWN_FACE: {} {} {}".format(img_name, face['distance'], face['id']))


	def run(self):
		frame_interval = 30  # Number of frames after which to run face detection
		fps_display_interval = 5  # seconds
		frame_rate = 0
		frame_count = 0

		self.recognizer = Recognizer()

		self.saved_faces = self.recognizer.get_saved_faces()

		cap = cv2.VideoCapture(os.path.join(DIRNAME, '../../video.webm'))
		#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
		#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
		#cap.set(cv2.CAP_PROP_FPS, 30)

		start_time = time.time()

		while True:
			detect_faces = None
			recognized_faces = None

			# Capture frame-by-frame
			ret, frame = cap.read()
			if (ret == 0):
				print("Error: check if webcam is connected.")
				return

			if ((frame_count % frame_interval) == 0):
				detected_faces = self.recognizer.detect_faces(frame)

				if detected_faces:
					recognized_faces = self.recognizer.recognize_faces(self.saved_faces, detected_faces)
					unrecognized_faces = [face for face in recognized_faces if face['recognized'] == False]

					if unrecognized_faces:
						self.alert_unrecognized_faces(unrecognized_faces, frame)
				# Check our current fps
				end_time = time.time()
				if (end_time - start_time) > fps_display_interval:
					frame_rate = int(frame_count / (end_time - start_time))
					start_time = time.time()
					frame_count = 0

			#if detected_faces is not None:
			#	print('painting detected')
			#	self.recognizer.add_overlays(frame, recognized_faces or detected_faces, frame_rate)

			frame_count += 1

			if (os.getenv('DEBUG') == True):
				cv2.imshow(self.window_name, frame)

			keyPressed = cv2.waitKey(1) & 0xFF
			if keyPressed == 27: # ESC key
					break
			elif keyPressed == 13: # ENTER key
					cv2.imwrite(self.window_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg", frame)
					print('Screenshot saved!')
		# When everything is done, release the capture
		cap.release()
		cv2.destroyAllWindows()
