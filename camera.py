import requests

class CameraConfig:
    def __init__(self):
        self.commandUrl = 'http://192.168.2.167/PSIA/YG/PTZCtrl/channels/0/continuous'

        self.stop_url = self.command_url + '?pan=0&tilt=0'
        self.right_url = self.command_url + '?pan=-1&tilt=0'
        self.left_url = self.command_url + '?pan=1&tilt=0'
        self.up_url = self.command_url + '?pan=0&tilt=1'
        self.down_url = self.command_url + '?pan=0&tilt=-1'

        self.streamUrl = 'rtsp://192.168.2.167:8554/profile1' 


    def sendCommand(self, command_url):        
        try:
            response = requests.put(command_url)
        except requests.exceptions.RequestException as error:
            print(error)

    def turnLeft(self):
        self.sendCommand(self.left_url)
     
    def turnRight(self):
        self.sendCommand(self.right_url)
     
    def stop(self):
        self.sendCommand(self.stop_url)
     