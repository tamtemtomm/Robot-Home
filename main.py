from depth_camera.tools import DepthCamera
from depth_camera.display import App
from openRb.arm_controller.controller import ARM_ROBOT


if __name__ == '__main__':
    
    robot = ARM_ROBOT()
    
    cam = DepthCamera(
        cam=0,
        bracket_theta=30)
    
    app = App(cam, robot=robot)
    app.run(verbose=False)