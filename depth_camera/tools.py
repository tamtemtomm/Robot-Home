import cv2
import numpy as np
import torch
from primesense import openni2
from primesense import _openni2 as c_api
import os, shutil, time, json
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import threading

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from config import *

class DepthCamera :
    def __init__(self, 
                 cam = 1,
                 redist = REDIST_PATH,
                 data_dir = DATA_DIR,
                 thread_progress=True):
        self.cam = cam
        self.redist = redist
        self.data_dir = data_dir
        self.device = device
        
        self.thread_progress = thread_progress
        if self.thread_progress:
            self.thread = threading.Thread(target=self.get_frame, daemon=True, args=(False,))
    
    def config(self, 
                depth=True,
                color=True,
                yolo=True,
                temporal_filter=False,
                colormap=False,
                width=640,
                height=480,
                fps=30,
                save_data = False):
        self.width = width
        self.height = height
        self.color = color
        self.depth = depth
        
        if self._check_params(depth,
            color,
            yolo,
            temporal_filter,
            colormap,
            width,
            height,
            fps,
            save_data) :
            return 
        
        ##---------------------------------------------------------------------------------------------------------
        # YOLO MODEL INITIALISATION
        
        self.model = self._yoloModel()
        
        ##---------------------------------------------------------------------------------------------------------
        # SAVE DATA INITIALITATION
        
        if save_data:
            self.camera_data = CameraData(self.data_dir, 
                                     color=color, 
                                     depth=depth, 
                                     depth_data=depth)
        
        ##---------------------------------------------------------------------------------------------------------
        # DEPTH STREAM INITIALITATION
        if depth:
            self.depth_stream = DepthStream(redist=self.redist, 
                                       width=width, 
                                       height=height, 
                                       fps=fps)
        else : self.depth_image = None
        
        ##---------------------------------------------------------------------------------------------------------
        # COLOR STREAM INITIALITATION
        if color:
            self.color_stream = ColorStream(cam=self.cam)
        else : self.color_image = None
        
    def run(self):
        
        if self.thread_progress:
            self.thread.start()
        else : 
            self.loop()
            
        self.close()
        
    ##-----------------------------------------------------------------------------------------------
    ## Add Extra Functions
    
    def close(self):
        if self.depth:
            self.depth_stream.close()
        if self.color:
            self.color_stream.close()
    
    def loop(self, show=True):
         while True :
            
            self.depth_image, self.img_depth, self.color_image = self.get_frame(show=show)
            
            ##----------------------------------------------------------------------------------------------------
            # SAVING DATA
            if self.save_data:
                self.camera_data.save(self.color_image, self.depth_image, self.img_depth)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    
    def get_frame(self, show=False):
        ##----------------------------------------------------------------------------------------------------
        # GET DEPTH FRAME 
        if self.depth:
            self.depth_image, self.img_depth = self.depth_stream.get_frame(colormap=self.colormap,
                                                                           temporal_filter=self.temporal_filter)
            if show: 
                cv2.imshow('Depth Image', self.depth_image)
        
        else :
            self.depth_image = None
            self.img_depth = None
            
        ##----------------------------------------------------------------------------------------------------
        # GET COLOR FRAME 
        if self.color:       
            self.color_image = self.color_stream.get_frame(img_depth=self.img_depth, 
                                                           model=self.model if self.yolo else None,
                                                           temporal_filter=self.temporal_filter)
            if show: 
                cv2.imshow("Color Image", self.color_image)
        
        else : self.color_image = None
        
        return self.depth_image, self.img_depth, self.color_image
    
    def _yoloModel(self):
        model = YOLO(YOLO_SEG_MODEL_PATH)
        return model.to(self.device)
    
    def _check_params( self,
            depth,
            color,
            yolo,
            temporal_filter,
            colormap,
            width,
            height,
            fps,
            save_data):
        
        for args in [depth, color, yolo, colormap, temporal_filter, save_data]:
            if isinstance(args, bool) == False:
                print(f'(depth, color, yolo, colormap, temporal_filter, save_data) parameter only accept boolean datatype')
                return True
        for args in [width, height, fps]:
            if isinstance(args, int) == False:
                print(f'(width, height, fps) parameter only accept int datatype')
                return True
            
        self.yolo = yolo
        self.color = color
        self.depth = depth
        self.depth_data = depth
        self.temporal_filter = temporal_filter
        self.colormap = colormap
        self.save_data = save_data
        return False


class DepthStream :
    def __init__(self, redist, width, height, fps):
        openni2.initialize(redist)
        dev = openni2.Device.open_any()
        self.depth_stream = dev.create_depth_stream()
        self.depth_stream.start()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = width, resolutionY = height, fps = fps))
    
    def get_frame(self, colormap=False, temporal_filter=False):
        self.temporal_filter = temporal_filter
        if self.temporal_filter:
            self.prev_depth_image = None
        
        depth_frame = self.depth_stream.read_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        img_depth = np.frombuffer(depth_frame_data, dtype=np.uint16).astype(np.float32)
        img_depth.shape = (480, 640)
        depth_image = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if colormap:
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        
        if self.temporal_filter :
            depth_image = self._temporal_filter(depth_image, self.prev_depth_image)
            self.prev_depth_image = depth_image
        
        return depth_image, img_depth
    
    def _temporal_filter(self, frame, prev_frame=None, alpha=0.5):
        if prev_frame is None : 
            return frame
        else : 
            result = cv2.addWeighted(frame, alpha, prev_frame, 1-alpha, 0)
            return result
    
    def close(self):
        openni2.unload()

class ColorStream:
    def __init__(self, cam):
        self.cap = cv2.VideoCapture(cam)
            
    def get_frame(self, img_depth=None, model=None, temporal_filter=False):
        self.temporal_filter = temporal_filter
        if self.temporal_filter:
            self.prev_color_image = None
        
        _, color_image = self.cap.read()
        
        if model is not None:
            self.model = model
            color_image = self._yolo(color_image, img_depth)
        
        if self.temporal_filter :
            color_image = self._temporal_filter(color_image, self.prev_color_image)
            self.prev_color_image = color_image
        
        return color_image
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _yolo(self, img, img_depth=None):
        annotator = Annotator(img)
        results = self.model.predict(img, verbose=False)
        if results:
            for r in results:
                if r.boxes and r.masks: 
                    for box, mask in zip(r.boxes, r.masks):
                        img, annotator = self._annotate_segment(img, box, mask, annotator, img_depth)
                            
        img = annotator.result() 
        return img
    
    def _annotate_segment(self, img, box, mask, annotator, img_depth) : 
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
        mask_segment = mask.data.to(device).numpy()
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

if __name__ == '__main__':
    cam = DepthCamera(cam=0)
    cam.config(depth=False, yolo=False)
    cam.run()