import cv2
import numpy as np
from .face_detector.FaceDetector import FaceDetector

class CaptureUser:
	def __init__(self, ImageCreator):
		self.detector = FaceDetector()
		self.image_creator = ImageCreator()
		self.stream_url = 'rtsp://192.168.2.167:8554/profile1'

	def run(self):
		frame_interval = 30  # Number of frames after which to run face detection
		fps_display_interval = 5  # seconds
		frame_rate = 0
		frame_count = 0

		amostra = 1
		numeroAmostra = 25
		largura, altura = 220, 220
		id = input('Digite o seu identificador: ')

		ca = cv2.VideoCapture(self.stream_url)
		print("Capturando as faces.....")

		start_time = time.time()

		while True:
			connected, frame = camera.read()

			if (connected is True):
				if ((frame_count % frame_interval) == 0):
					detected_faces = self.detector.detect_faces(frame)

					if (detected_faces):
						print(detected_faces)
						# adiciona contorno na face detectada/reconhecida e nome caso seja reconhecida
						self.add_overlays(frame, detected_faces, frame_rate)

					"""
					for (x, y, l, a) in detected_faces:
						cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
						regiao = frame[y:y + a, x:x + l]
						regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
						olhosDetectados = classificadorOlhos.detectMultiScale(
							regiaoCinzaOlho)
						for (ox, oy, ol, oa) in olhosDetectados:
							cv2.rectangle(regiao, (ox, oy),
										(ox + ol, oy + oa), (0, 255, 0), 2)
							if cv2.waitKey(1) & 0xFF == ord('q'):
								if np.average(gray_frame) > 110:
									frameFace = cv2.resize(
										gray_frame[y:y + a, x:x + l], (largura, altura))
									cv2.imwrite("fotos/pessoa." + str(id) +
												"." + str(amostra) + ".jpg", frameFace)
									print("[foto " + str(amostra) +
										" capturada com sucesso")
									amostra += 1
					"""
					# Check our current fps
					end_time = time.time()
					if (end_time - start_time) > fps_display_interval:
						frame_rate = int(frame_count / (end_time - start_time))
						start_time = time.time()
						frame_count = 0

				frame_count += 1

				cv2.imshow("Face", frame)
		# cv2.waitKey(1)
			if (amostra >= numeroAmostra + 1):
				break
		print("Faces capturadas com sucesso")
		camera.release()
		cv2.destroyAllWindows()
