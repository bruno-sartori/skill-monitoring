from camera import Camera
from time import sleep
from movement import Movement

class Monitoring:

    def run(self):
        try:
            self.t1 = Camera()
            sleep(10)
            self.t2 = Movement()
        except Exception as e:
            print(e)
        
        try:
            self.t1.join()
            self.t2.join()
        except Exception as e:
            print(e)

if __name__ == '__main__':
    m = Monitoring()
    m.run()