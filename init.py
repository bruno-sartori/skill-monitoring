import argparse
import sys
from monitoring.camera.CameraManagement import CameraManagement
from monitoring.camera.ImageCreator import ImageCreator
from monitoring.modules.facenet.recognizer import Recognizer as FacenetRecognizer
from monitoring.modules.tracker.camera import Camera as Tracker
from monitoring.api.api import startAPI

parser = argparse.ArgumentParser()
parser.add_argument('--module', help='Module to execute: facenet|tracker')
parser.add_argument('--host', help='API Host')
parser.add_argument('--port', help='API port')

args = parser.parse_args()

def track():
	t = Tracker(direction=True)
	t.run()

def recognizeFacenet():
	c = CameraManagement(FacenetRecognizer, ImageCreator)
	c.run()

def api():
	host = args.host if args.host != '' else 'localhost'
	port = args.port if args.port != '' else 3000
	startAPI(host, port)

def execute(module):
	switcher = {
		"tracker": track,
		"facenet": recognizeFacenet,
		"api": api
	}
	func = switcher.get(module, lambda: print("Invalid module {}".format(module)))
	func()


if __name__ == '__main__':
	execute(args.module)
