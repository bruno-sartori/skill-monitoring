from camera import Camera
from movement import Movement

class Monitoring:

    def run(self):
        try:
            self.t1 = Movement()
            self.t2 = Camera()
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