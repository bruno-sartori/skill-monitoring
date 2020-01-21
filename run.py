from monitoring.camera.CameraManagement import CameraManagement
from monitoring.camera.ImageCreator import ImageCreator
from monitoring.modules.facenet.recognizer import Recognizer

if __name__ == '__main__':
    c = CameraManagement(Recognizer, ImageCreator)
    c.run()
