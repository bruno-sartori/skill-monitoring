import copyreg as copy_reg
from joblib import Parallel, delayed
import multiprocessing
import sys
import time
import os, os.path
import datetime
import tensorflow as tf
import numpy as np
from monitoring.modules.facenet import facenet
from monitoring.align import detect_face
import cv2
import argparse
from joblib import Parallel, delayed
import multiprocessing
import types
from monitoring.utils.constants import ERROR, SUCCESS

DIRNAME = os.path.dirname(os.path.abspath(__file__))
UNKNOWN_FACE = 'Unknown'
NUM_CORE = multiprocessing.cpu_count()

class Recognizer:
	def __init__(self):
		# some constants kept as default from facenet
		self.minsize = 20
		self.threshold = [0.6, 0.7, 0.7]
		self.factor = 0.709
		self.margin = 44
		self.input_image_size = 160
		self.window_name="teste"

		# session
		self.sess = tf.Session()

		# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
		pnet, rnet, onet = detect_face.create_mtcnn(self.sess, False)
		self.pnet = pnet
		self.rnet = rnet
		self.onet = onet

		# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk

		facenet.load_model(os.path.join(DIRNAME, "./models/20170512-110547/20170512-110547.pb"))

		# Get input and output tensors
		self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		self.embedding_size = self.embeddings.get_shape()[1]

	def setup(self):
		self.saved_faces = self.get_saved_faces()


	def get_face_data(self, face, img_size, img):
		if face[4] > 0.50:
			det = np.squeeze(face[0:4])

			bb = np.zeros(4, dtype=np.int32)
			bb[0] = np.maximum(det[0] - self.margin / 2, 0)
			bb[1] = np.maximum(det[1] - self.margin / 2, 0)
			bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
			bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])

			cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
			resized = cv2.resize(cropped, (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
			prewhitened = facenet.prewhiten(resized)
			embedding = self.getEmbedding(prewhitened)

			return {'face': resized, 'cropped': cropped, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'embedding': embedding, 'name': None }
		raise ValueError('face[4] must be > 0.50')

	def detect_faces(self, img):
		faces = []
		img_size = np.asarray(img.shape)[0:2]
		bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

		if not len(bounding_boxes) == 0:
			for face in bounding_boxes:
				data = self.get_face_data(face, img_size, img)
				faces.append(data)
		return faces


	def recognize_faces(self, found_faces):
		threshold = 0.9

		processed_faces = []

		for found_face in found_faces:
			recognized_face = { 'dist': 100, 'face': None }
			unrecognized_face = { 'dist': 100, 'face': None }

			for saved_face in self.saved_faces:
				compared = self.compare_faces(saved_face, found_face, threshold)

				if (compared['type'] == SUCCESS and compared['dist'] < recognized_face['dist']):
					recognized_face['dist'] = compared['dist']
					recognized_face['face'] = saved_face
				elif (compared['type'] == ERROR and recognized_face['face'] is None):
    				# find best predicted wrong face
					if (compared['dist'] < unrecognized_face['dist']):
						unrecognized_face['dist'] = compared['dist']
						unrecognized_face['face'] = saved_face

			if recognized_face['face'] is not None:
				print('RECOGNIZED: {0} | DISTANCE: {1}'.format(recognized_face['face']['name'], recognized_face['dist']))
				found_face['recognized'] = True
				found_face['distance'] = recognized_face['dist']
				found_face['name'] = recognized_face['face']['name']
				found_face['id'] = recognized_face['face']['id']
				processed_faces.append(found_face)
			elif unrecognized_face['face'] is not None:
				found_face['recognized'] = False
				found_face['distance'] = unrecognized_face['dist']
				found_face['name'] = unrecognized_face['face']['name']
				found_face['id'] = unrecognized_face['face']['id']
				processed_faces.append(found_face)
			else:
				found_face['recognized'] = False
				found_face['distance'] = None
				found_face['name'] = UNKNOWN_FACE
				found_face['id'] = None
				processed_faces.append(found_face)
		return processed_faces


	def add_overlays(self, frame, faces, frame_rate):
		if faces is not None:
			for face in faces:
				bb = face['rect']
				cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
				if face['name'] is not None:
					cv2.putText(frame, face['name'], (bb[0], bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

		cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)


	def getEmbedding(self, resized):
		reshaped = resized.reshape(-1, self.input_image_size, self.input_image_size,3)
		feed_dict = { self.images_placeholder: reshaped, self.phase_train_placeholder: False }
		embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)
		return embedding

	def compare_faces(self, face1, face2, threshold):
		if face1 and face2:
			# calculate Euclidean distance
			dist = np.sqrt(np.sum(np.square(np.subtract(face1['embedding'], face2['embedding']))))

			if (dist <= threshold):
				return { 'dist': dist, 'type': SUCCESS }
			return { 'dist': dist, 'type': ERROR }
		raise ValueError('all faces must be passed')

	def get_saved_faces(self):
		start_time = time.time()
		saved_faces = []

		for file in os.listdir(os.path.join(DIRNAME, './images')):
			print("FILE: ", file)
			img = cv2.imread(os.path.join(DIRNAME, './images/', file))
			face = self.detect_faces(img)
			if face:
				saved_face = face[0]
				saved_face['id'] = file.split('_')[0]
				saved_face['name'] = file.split('_')[1]
				saved_faces.append(saved_face)

		end_time = time.time() - start_time
		print("read {0} faces in {1}s".format(len(saved_faces), end_time))
		return saved_faces
