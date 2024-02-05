from tools import DepthCamera
from config import *
from display import App

if __name__ == '__main__':

    # Change cam parameter based on your camera
    # thread_progress = False to unable threading function
    cam = DepthCamera(cam = 0)
    
    # .config parameters :

    # 1. yolo            : True (Default) | False
    # 2. temporal_filter : True (Default) | False
    # 3. colormap        : True           | False (Default)
    # 4. save_data       : True           | False (Default)

    cam.config()
    
    app = App(cam)
    app.run(verbose=True)