from monitoring.camera.face_detector import MTCNNFaceAlignment as detect_face
import numpy as np
import tensorflow as tf
import cv2

class FaceDetector:
	def __init__(self):
		# some constants kept as default from facenet
		self.minsize = 20
		self.threshold = [0.6, 0.7, 0.7]
		self.factor = 0.709
		self.margin = 44
		self.input_image_size = 160

		# session
		self.sess = tf.Session()

		# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
		pnet, rnet, onet = detect_face.create_mtcnn(self.sess, False)
		self.pnet = pnet
		self.rnet = rnet
		self.onet = onet

	def detect_faces(self, img):
		faces = []
		img_size = np.asarray(img.shape)[0:2]
		bounding_boxes, _ = detect_face.detect_face(
			img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

		if not len(bounding_boxes) == 0:
			for face in bounding_boxes:
				data = self.get_face_data(face, img_size, img)
				faces.append(data)
		return faces

	def get_face_data(self, face, img_size, img):
		if face[4] > 0.50:
			det = np.squeeze(face[0:4])

			bb = np.zeros(4, dtype=np.int32)
			bb[0] = np.maximum(det[0] - self.margin / 2, 0)
			bb[1] = np.maximum(det[1] - self.margin / 2, 0)
			bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
			bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])

			cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
			resized = cv2.resize(cropped, (self.input_image_size,
                                  self.input_image_size), interpolation=cv2.INTER_CUBIC)

			return { 'face': resized, 'cropped': cropped, 'rect': [bb[0], bb[1], bb[2], bb[3]], 'name': None }
		raise ValueError('face[4] must be > 0.50')
