from depth_camera.tools import DepthCamera
from depth_camera.display import App
from openRb.arm_controller.controller import ARM_ROBOT


if __name__ == '__main__':
    
    robot = ARM_ROBOT(com_port="COM7")
    
    cam = DepthCamera(
        cam=3,
        bracket_theta=30)
    
    app = App(cam, robot=robot)
    app.run(verbose=False)