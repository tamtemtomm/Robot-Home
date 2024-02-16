from depth_camera.tools import DepthCamera
from depth_camera.config import *
from depth_camera.display import App

if __name__ == '__main__':

    # Change cam parameter based on your camera
    # thread_progress = False to unable threading function
    cam = DepthCamera(cam = 0)
    
    # .config parameters :

    # 1. yolo            : True (Default) | False
    # 2. temporal_filter : True (Default) | False
    # 3. colormap        : True           | False (Default)
    # 4. save_data       : True           | False (Default)

    app = App(cam)
    app.run(verbose=False)