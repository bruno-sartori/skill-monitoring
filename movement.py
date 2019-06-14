from threading import Thread
from time import sleep
import requests

class Movement(Thread):
    def __init__(self):
        print("INITIALIZING MOVIMENT...")
        Thread.__init__(self)
        self.daemon = True
        self.running = True
    
        self.command_url = 'http://192.168.2.167/PSIA/YG/PTZCtrl/channels/0/continuous'

        self.stop_url = self.command_url + '?pan=0&tilt=0'
        self.right_url = self.command_url + '?pan=-1&tilt=0'
        self.left_url = self.command_url + '?pan=1&tilt=0'
        self.up_url = self.command_url + '?pan=0&tilt=1'
        self.down_url = self.command_url + '?pan=0&tilt=-1'
        
        self.start()

    def run(self):
        while self.running:
            print("MOVING LEFT")
            self.turnLeft()
            sleep(2)
            self.stop()
            sleep(4)
            print("MOVING LEFT")
            self.turnLeft()
            sleep(2)
            self.stop()
            sleep(4)
            print("MOVING LEFT")
            self.turnLeft()
            sleep(2)
            self.stop()
            sleep(4)
            
            print("MOVING RIGHT")
            self.turnRight()
            sleep(2)
            self.stop()
            sleep(4)
            print("MOVING RIGHT")
            self.turnRight()
            sleep(2)
            self.stop()
            sleep(4)
            print("MOVING RIGHT")
            self.turnRight()
            sleep(2)
            self.stop()
            sleep(4)
    
    
    def stop(self):
        self.stop()
        self.running = False
    

    def sendCommand(self, command_url):
        try:
            response = requests.put(command_url)
            print(response)
        except requests.exceptions.RequestException as error:
            print(error)

    def turnLeft(self):
        self.sendCommand(self.left_url)
     
    def turnRight(self):
        self.sendCommand(self.right_url)
     
    def stop(self):
        self.sendCommand(self.stop_url)
