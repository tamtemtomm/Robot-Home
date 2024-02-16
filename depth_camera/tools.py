from depth_camera.config import *
from depth_camera.colorstream import ColorStream
from depth_camera.depthstream import DepthStream
from depth_camera.cameradata import CameraData

import cv2
import torch
from primesense import openni2
from ultralytics import YOLO
import threading
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DepthCamera :
    def __init__(self, 
                cam = 1,
                robot=None,
                redist = REDIST_PATH,
                data_dir = DATA_DIR,
                thread_progress=True,
                bracket_theta=30,
                height=200,
                yolo=True,
                temporal_filter=False,
                colormap=False,
                save_data = False):
        
        self.cam = cam
        self.robot = robot
        self.redist = redist
        self.data_dir = data_dir
        self.device = device
        self.focal_length = 60
        self.height = height
        self.bracket_theta = bracket_theta
        self.width = 640
        self.height = 480
        self.fps = 30
        self.yolo = yolo
        self.temporal_filter = temporal_filter
        self.colormap = colormap
        self.save_data = save_data
        
        self.thread_progress = thread_progress
        if self.thread_progress:
            self.thread = threading.Thread(target=self.get_frame, daemon=True, args=(False,))
            
        openni2.initialize(self.redist)
        if openni2.Device.open_all() == []:
            self.depth=False
            self.depth_data=False
        else :
            self.depth=True
            self.depth_data=True
        
        self.data = CameraData(bracket_theta = self.bracket_theta)
        
        self.model, self.gripper_model, self.barcode_model = self._model()
        
        ##---------------------------------------------------------------------------------------------------------
        # DEPTH STREAM INITIALITATION
        if self.depth:
            self.depth_stream = DepthStream(redist=self.redist, 
                                            width=self.width, 
                                            height=self.height, 
                                            fps=self.fps)
        else : self.depth_image = None
        
        ##---------------------------------------------------------------------------------------------------------
        # COLOR STREAM INITIALITATION
        self.color_stream = ColorStream(cam=self.cam)
    
    def run(self, 
            verbose=False):
        
        if self.thread_progress:
            self.thread.start()
        
        self.loop(verbose=verbose)
            
        self.close()
        
    
    def close(self):

        if self.depth:
            self.depth_stream.close()
        self.color_stream.close()
    
    def loop(self, show=True, verbose=False, robot=None):
        if self.save_data:
            self.data.setup()
        
        while True :
            self.depth_image, self.img_depth, self.color_image = self.get_frame(show=show, verbose=verbose)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if self.save_data:
            self.data.save()
        
    
    def get_frame(self, 
                  show=False, 
                  verbose=False):
        ##----------------------------------------------------------------------------------------------------
        # GET DEPTH FRAME 
        
        self.cur_data = self.data.set_data()
        
        if self.depth:
            self.depth_image, self.img_depth, self.cur_data = self.depth_stream.get_frame(
                                                                colormap=self.colormap,
                                                                temporal_filter=self.temporal_filter,
                                                                data = self.cur_data)
            
            if self.depth_image is not None:
                if show: 
                    cv2.imshow('Depth Image', self.depth_image)
        
        else :
            self.depth_image = None
            self.img_depth = None
            
        ##----------------------------------------------------------------------------------------------------
        # GET COLOR FRAME       

        self.color_image, self.cur_data = self.color_stream.get_frame(
                                                            img_depth=self.img_depth, 
                                                            model=self.model if self.model else None,
                                                            gripper_model=self.gripper_model if self.gripper_model else None,
                                                            barcode_model=self.barcode_model,
                                                            temporal_filter=self.temporal_filter,
                                                            data = self.cur_data)
        if self.color_image is not None:
            if show: 
                cv2.imshow("Color Image", self.color_image)

        if verbose:
            print(self.cur_data)
        
        if self.robot is not None:
            x, y, z = self.data.cur_process()['min']['item_location']
            print(x, y, z)
            # robot.inverse_kinematics(x, y, z)
        
        print(self.data.cur_process())

        self.data.append(self.cur_data, save=self.save_data)
        
        return self.depth_image, self.img_depth, self.color_image
    
    def _model(self):
        try:
            model = YOLO(YOLO_SEG_MODEL_PATH)
            model = model.to(self.device)
        except:
            model = None
        
        try:
            gripper_model = YOLO(YOLO_GRIPPER_MODEL_PATH)
            gripper_model = gripper_model.to(self.device)
        except:
            gripper_model = None
        
        try:
            barcode_model = YOLO(YOLO_GRIPPER_MODEL_PATH)
            barcode_model = barcode_model.to(self.device)
        except:
            barcode_model = None

        return model, gripper_model, barcode_model

    # def _check_params( self,
    #         yolo,
    #         temporal_filter,
    #         colormap,
    #         save_data):
        
    #     # for args in [depth, color, yolo, colormap, temporal_filter, save_data]:
    #     #     if isinstance(args, bool) == False:
    #     #         print(f'(depth, color, yolo, colormap, temporal_filter, save_data) parameter only accept boolean datatype')
    #     #         return True
    #     # for args in [width, height, fps]:
    #     #     if isinstance(args, int) == False:
    #     #         print(f'(width, height, fps) parameter only accept int datatype')
    #     #         return True
            
    #     self.yolo = yolo
    #     self.temporal_filter = temporal_filter
    #     self.colormap = colormap
    #     self.save_data = save_data
    #     return False

                        
if __name__ == '__main__':
    
    cam = DepthCamera(cam=0,
                      robot=1)
    # cam.config()
    cam.run(verbose=True)