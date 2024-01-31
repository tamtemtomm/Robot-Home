from tools import DepthCamera
from config import *
from display import App

if __name__ == '__main__':

    # Change cam parameter based on your camera
    # thread_progress = False to unable threading function
    cam = DepthCamera(cam = 1)
    
    # .config parameters :
    # 1. depth           : True (Default) | False 
    # 2. color           : True (Default) | False
    # 3. yolo            : True (Default) | False
    # 4. temporal_filter : True (Default) | False
    # 5. width           : 640 (DEFAULT)
    # 6. height          : 480 (DEFAULT)
    # 7. fps             : 30  (DEFAULT)
    # 8. save_data       : True           | False(Default)

    cam.config()
    
    app = App(cam)
    app.run()