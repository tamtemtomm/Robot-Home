from depth_camera.tools import DepthCamera
from depth_camera.display import App
from openRb.arm_controller.controller import ARM_ROBOT


if __name__ == '__main__':
    cam = DepthCamera(cam=0)
    cam.config()
        
    app = App(cam)
    
    while True:
        