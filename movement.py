from threading import Thread
from time import sleep
import requests

class Movement():
    def __init__(self):
        print("INITIALIZING MOVIMENT...")
        self.running = True
    
        self.command_url = 'http://192.168.2.167/PSIA/YG/PTZCtrl/channels/0/continuous'

        self.direction_command = {
            'RIGHT': self.command_url + '?pan=-1&tilt=0',
            'LEFT': self.command_url + '?pan=1&tilt=0',
            'UP': self.command_url + '?pan=0&tilt=1',
            'DOWN': self.command_url + '?pan=0&tilt=-1',
            'UP-LEFT': self.command_url + '?pan=1&tilt=1',
            'UP-RIGHT': self.command_url + '?pan=-1&tilt=1',
            'DOWN-LEFT': self.command_url + '?pan=1&tilt=-1',
            'DOWN-RIGHT': self.command_url + '?pan=-1&tilt=-1',
            'STOP': self.command_url + '?pan=0&tilt=0'
        }
        
        
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
        print("COMMAND SEND")
        """
        try:
            response = requests.put(command_url)
            print(response)
        except requests.exceptions.RequestException as error:
            print(error)
        """

    def move(self, direction):
        print("MOVING:  {}".format(direction))
        if (direction != ''):
            try:
                response = requests.put(self.direction_command[direction])
                print(response)
            except requests.exceptions.RequestException as error:
                print(error)

    def stop(self):
        self.sendCommand(self.stop_url)
