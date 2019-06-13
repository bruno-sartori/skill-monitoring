# TODO: Add an appropriate license to your skill before publishing.  See
# the LICENSE file for more information.

# Below is the list of outside modules you'll be using in your skill.
# They might be built-in to Python, from mycroft-core or from external
# libraries.  If you use an external library, be sure to include it
# in the requirements.txt file so the library is installed properly
# when the skill gets installed later by a user.

from adapt.intent import IntentBuilder
from mycroft.skills.core import MycroftSkill, intent_handler
from mycroft.util.log import LOG
import cv2
import time
import numpy
import glob
import logging
from .detector import DetectorAPI
from .camera import CameraConfig
from threading import Thread
from time import sleep

def threaded(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper


# Each skill is contained within its own class, which inherits base methods
# from the MycroftSkill class.  You extend this class as shown below.

# TODO: Change "Monitoring" to a unique name for your skill
class MonitoringSkill(MycroftSkill):

    # The constructor of the skill, which calls MycroftSkill's constructor
    def __init__(self):
        super(MonitoringSkill, self).__init__(name="MonitoringSkill")
        
        # Initialize working variables used within the skill.
        self.lock = True
        self.detected = 0
        self.cameraConfig = CameraConfig()
        
    # The "handle_xxxx_intent" function is triggered by Mycroft when the
    # skill's intent is matched.  The intent is defined by the IntentBuilder()
    # pieces, and is triggered when the user's utterance matches the pattern
    # defined by the keywords.  In this case, the match occurs when one word
    # is found from each of the files:
    #    vocab/en-us/Hello.voc
    #    vocab/en-us/World.voc
    # In this example that means it would match on utterances like:
    #   'Hello world'
    #   'Howdy you great big world'
    #   'Greetings planet earth'
    #@intent_handler(IntentBuilder("").require("Hello").require("World"))
    #def handle_hello_world_intent(self, message):
        # In this case, respond by simply speaking a canned response.
        # Mycroft will randomly speak one of the lines from the file
        #    dialogs/en-us/hello.world.dialog
    #    self.speak_dialog("hello.world")

    def personDetected(self, img):
        logging.error("PERSON DETECTED")

    @threaded
    def detect(self, img):
        model_path = '/home/bruno/Documentos/Projetos/octopus/skill_monitoring/ssd_mobilenet_v1_coco/frozen_inference_graph.pb'
        odapi = DetectorAPI(path_to_ckpt=model_path)
        cap = cv2.VideoCapture(self.cameraConfig.stream_url)
        threshold = 0.7

        while self.lock:
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
                              # Read image
            img = cv2.resize(img, (960, 540))
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                self.stop()

    @threaded
    def moviment():
        logging.debug("LEFT")
        self.cameraConfig.turnLeft()
        sleep(2)
        self.cameraConfig.stop()
        self.sleep(1)
        logging.debug("RIGHT")
        self.cameraConfig.turnRight()
        self.sleep(2)
        self.moviment()

    @intent_handler(IntentBuilder("").require("Monitoring"))
    def handle_monitoring_intent(self, message):
        self.speak_dialog("monitoring.started", data={"date": "TESTE" })
        
        self.moviment()
        self.detect()

        print("thread finished...exiting")


    def stop(self):
        self.lock = False
        return True

    # The "stop" method defines what Mycroft does when told to stop during
    # the skill's execution. In this case, since the skill's functionality
    # is extremely simple, there is no need to override it.  If you DO
    # need to implement stop, you should return True to indicate you handled
    # it.
    #
    # def stop(self):
    #    return False

# The "create_skill()" method is used to create an instance of the skill.
# Note that it's outside the class itself.
def create_skill():
    return MonitoringSkill()
