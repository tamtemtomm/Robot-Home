import cv2
import numpy as np
import torch
from primesense import openni2
from primesense import _openni2 as c_api
import os, shutil, time, json
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from config import *

class DepthCamera :
    def __init__(self, 
                 cam = 1,
                 redist = REDIST_PATH,
                 data_dir = DATA_DIR):
        self.cam = cam
        self.redist = redist
        self.data_dir = data_dir
        self.previous_frame = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
    def run(self,
            depth=True,
            color=True,
            yolo=True,
            temporal_filter=False,
            width=640,
            height=480,
            fps=30,
            save_data = True,
            ):
        
        ##---------------------------------------------------------------------------------------------------------
        # PARAMETER CHECKING
        self.height = height
        self.depth = depth
        
        if depth == False:
            img_depth = None
            depth_image = None
        if color == False:
            color_image = None
        
        for args in [depth, color, save_data]:
            if isinstance(args, bool) == False:
                print(f'(depth, color, depth_data, blobs, save_data) parameter only accept boolean datatype')
                return         
        for args in [width, height, fps]:
            if isinstance(args, int) == False:
                print(f'(width, height, fps) parameter only accept int datatype')
                return
        ##---------------------------------------------------------------------------------------------------------
        # YOLO MODEL INITIALISATION
        
        if yolo:
            self.model = self._yoloModel()
        
        ##---------------------------------------------------------------------------------------------------------
        # SAVE DATA INITIALITATION
        
        if save_data:
            camera_data = CameraData(self.data_dir, color=color, depth=depth, depth_data=depth)
        
        ##---------------------------------------------------------------------------------------------------------
        # DEPTH STREAM INITIALITATION
        if depth:
            openni2.initialize(self.redist)
            dev = openni2.Device.open_any()
            depth_stream = dev.create_depth_stream()
            depth_stream.start()
            depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = width, resolutionY = height, fps = fps))
        
        ##---------------------------------------------------------------------------------------------------------
        # COLOR STREAM INITIALITATION
        if color:
            cap = cv2.VideoCapture(self.cam)
            
        if temporal_filter:
            prev_color_image=None
            prev_depth_image=None
        
        i = 0
        
        while True :
            
            ##----------------------------------------------------------------------------------------------------
            # GET DEPTH FRAME 
            if depth:
                depth_frame = depth_stream.read_frame()
                depth_frame_data = depth_frame.get_buffer_as_uint16()
                img_depth = np.frombuffer(depth_frame_data, dtype=np.uint16).astype(np.float32)
                img_depth.shape = (480, 640)
                depth_image = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                
                if temporal_filter:
                    depth_image = self._temporal_filter(depth_image, prev_depth_image)
                    prev_depth_image = depth_image
                
                cv2.imshow('Depth Image', depth_image)
                
            ##----------------------------------------------------------------------------------------------------
            # GET COLOR FRAME 
            if color:
                _, color_image = cap.read()
                  
                if temporal_filter:
                    color_image = self._temporal_filter(color_image, prev_color_image)
                    prev_color_image = color_image

                ##--------------------------------------------------------------------------------------------------
                # GET YOLO ANNOTATION 
                
                if yolo:
                    annotator = Annotator(color_image)
                    
                    results = self.model.predict(color_image, verbose=False)
                    if results:
                        for r in results:
                            if r.boxes and r.masks: 
                                for box, mask in zip(r.boxes, r.masks):
                                    color_image, annotator = self._annotate_segment(color_image, box, mask, annotator, img_depth)
                                        
                    color_image = annotator.result()         
                
                cv2.imshow("Color Image", color_image,)
                            
            ##----------------------------------------------------------------------------------------------------
            # SAVING DATA
            if save_data:
                camera_data.save(color_image, depth_image, img_depth)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            i+=1
        
        openni2.unload()
        cap.release()
        cv2.destroyAllWindows()
        
    ##-----------------------------------------------------------------------------------------------
    ## Add Extra Functions
    
    def _yoloModel(self):
        model = YOLO(YOLO_SEG_MODEL_PATH)
        return model.to(self.device)
    
    def _annotate_segment(self, img, box, mask, annotator, img_depth=None):
        bbox = box.xyxy[0]
        class_name = box.cls
        border = mask.xy[0]
        
        annotator.box_label(bbox, self.model.names[int(class_name)])
        img = self._add_border(img, border)

        if img_depth is not None:
            img = self._add_distance_estimation(img, mask, img_depth, bbox)
        
        return img, annotator
    
    def _add_border(self, img, border):
        for a, b in border:
            img[int(b)-1, int(a)-1, 0] = 0
            img[int(b)-1, int(a)-1, 1] = 0
            img[int(b)-1, int(a)-1, 2] = 255
        return img
    
    def _add_distance_estimation(self, img, mask, img_depth, bbox):
        mask_segment = mask.data.to(self.device).numpy()
        mask_segment.shape = (480, 640)
        
        center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
        
        depth_mask = img_depth * mask_segment
        depth_estimation = int(np.sum(depth_mask)/np.sum(mask_segment)/10)
        img = cv2.putText(img, f'{depth_estimation} cm', center, DEFAULT_FONT,  
                0.4, (0, 0, 255), 1, DEFAULT_LINE)
        return img
        
    def _temporal_filter(self, frame, prev_frame=None, alpha=0.5):
        if prev_frame is None : 
            return frame
        else : 
            result = cv2.addWeighted(frame, alpha, prev_frame, 1-alpha, 0)
            return result
    
class CameraData :
    def __init__(self, data_dir, color=True, depth=True, depth_data=True):
        self.color = color
        self.depth = depth
        self.depth_data = depth_data
        
        t = time.localtime()
        self.current_time = f'waktu-{time.strftime("%Y-%m-%d %H-%M-%S", t)}'
        self.data_directory = os.path.join(data_dir, self.current_time)
        print(f'Saving data to the {self.data_directory}')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.data_directory, exist_ok=True)
        if self.color : 
            os.makedirs(os.path.join(self.data_directory, 'depth'))
        if self.depth:
            os.makedirs(os.path.join(self.data_directory, 'color'))
        if self.depth_data : 
            os.makedirs(os.path.join(self.data_directory, 'depth_data'))
        
        self.i = 0
    
    def save(self, color_image=None, depth_image=None, img_depth=None):
        if self.color:
            cv2.imwrite(os.path.join(self.data_directory, 'color', f'color_{self.i}')+'.jpg', color_image)
        if self.depth:
            cv2.imwrite(os.path.join(self.data_directory, 'depth', f'depth_{self.i}')+'.jpg', depth_image)
        if self.depth_data:
            with open(os.path.join(self.data_directory, 'depth_data', f'depth_data_{self.i}')+'.txt', 'w') as f:
                json.dump(img_depth.reshape(480, 640).tolist(), f)
        
        self.i += 1