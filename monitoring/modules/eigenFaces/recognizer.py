import cv2

class Recognizer:
	def __init__(self, detector):
		self.detector = detector

		self.recognizer = cv2.face.EigenFaceRecognizer_create()
		self.image_width = 220
		self.image_height = 220

	def setup(self):
		self.recognizer.read("classificadorEigen.yml")
		self.saved_faces = self.get_saved_faces()

	def get_saved_faces():
		return {
			1: 'Bruno',
			2: 'Karina',
			3: 'Sabrina',
			4: 'Wilian'
		}

	def recognize_faces(self, found_faces):
		threshold = 0.9

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=2.0, minSize=(30, 30))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (self.image_width, self.image_height))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = 'Bruno'
        if id == 2:
            nome = 'Karina'
        if id == 3:
            nome = 'Sabrina'
        if id == 4:
            nome = 'Wilian'
        cv2.putText(imagem, nome, (x,y + (a + 30)), font, 2, (0,0,255))
        cv2.putText(imagem, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.Release()
cv2.destroyAllWindows()

