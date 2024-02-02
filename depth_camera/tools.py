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
        
        self.data = CameraData()
    
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
        
        self.model, self.gripper_model = self._model()
        
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
        # print(f'Result : {self.data.get_data()}')

        if self.depth:
            self.depth_stream.close()
        if self.color:
            self.color_stream.close()
    
    def loop(self, show=True):
         while True :
            self.depth_image, self.img_depth, self.color_image = self.get_frame(show=show)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    
    def get_frame(self, show=False):
        ##----------------------------------------------------------------------------------------------------
        # GET DEPTH FRAME 
        
        self.cur_data = self.data.set_data()
        
        if self.depth:
            self.depth_image, self.img_depth, self.cur_data = self.depth_stream.get_frame(colormap=self.colormap,
                                                                           temporal_filter=self.temporal_filter,
                                                                           data = self.cur_data)
            if show: 
                cv2.imshow('Depth Image', self.depth_image)
        
        else :
            self.depth_image = None
            self.img_depth = None
            
        ##----------------------------------------------------------------------------------------------------
        # GET COLOR FRAME 
        if self.color:       
            self.color_image, self.cur_data = self.color_stream.get_frame(
                                                                img_depth=self.img_depth, 
                                                                model=self.model if self.yolo else None,
                                                                temporal_filter=self.temporal_filter,
                                                                data = self.cur_data)
            if show: 
                cv2.imshow("Color Image", self.color_image)

            print(self.cur_data)

            self.data.append(self.cur_data)

        else : self.color_image = None
        
        if self.save_data:
            self.data.save_current()
        
        return self.depth_image, self.img_depth, self.color_image
    
    def _model(self):
        try:
            model = YOLO(YOLO_SEG_MODEL_PATH)
            model = model.to(self.device)
        except:
            model = None
        
        # try:
        #     gripper_model = YOLO(YOLO_GRIPPER_MODEL_PATH)
        #     gripper_model = gripper_model.to(self.device)
        # except:
        #     gripper_model = None
        
        gripper_model = None

        return model, gripper_model

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
    
    def get_frame(self, colormap=False, temporal_filter=False, data=None):
     
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
           
        if data:
            self.data = data
            self.data['depth']['raw'] = img_depth
            self.data['depth']['image'] = depth_image

            return depth_image, img_depth, self.data
            
        return depth_image, img_depth, None
    
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
            
    def get_frame(self, 
                  img_depth=None, 
                  model=None, 
                  temporal_filter=False,
                  data = None
                ):
        
        if data :
            self.data = data

        self.temporal_filter = temporal_filter
        if self.temporal_filter:
            self.prev_color_image = None
        
        _, color_image = self.cap.read()
        
        if data:
            self.data['color']['raw'] = color_image
        
        if model is not None:
            self.model = model
            color_image = self._yolo(color_image, img_depth)
        
        if self.temporal_filter :
            color_image = self._temporal_filter(color_image, self.prev_color_image)
            self.prev_color_image = color_image
        
        if data : 
            self.data['color']['annot'] = color_image
            return color_image, self.data
        
        return color_image, None
    
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
        class_name = self.model.names[int(box.cls)]
        border = mask.xy[0]
        
        annotator.box_label(bbox, class_name)
        img = self._add_border(img, border)
        
        mask_segment = mask.data.to(device).numpy()
        mask_segment.shape = (480, 640)

        # img = self._add_distance_estimation(img, mask, img_depth, bbox)
        if img_depth is not None:
            depth_mask = img_depth * mask_segment
            depth_estimation = int(np.sum(depth_mask)/np.sum(mask_segment)/10)
        else :
            mask_segment = None
            depth_estimation = 0

        center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
        location = (center[0], center[1], depth_estimation)
        
        img = cv2.putText(img, f'{location}', center, DEFAULT_FONT,  
                0.4, (0, 0, 255), 1, DEFAULT_LINE)
        
        self.data['items_loc'][class_name] = []
        self.data['items_loc'][class_name].append((bbox.numpy(), mask_segment))
        
        return img, annotator

    def _add_border(self, img, border):
        for a, b in border:
            img[int(b)-1, int(a)-1, 0] = 0
            img[int(b)-1, int(a)-1, 1] = 0
            img[int(b)-1, int(a)-1, 2] = 255
        return img
    
    def _temporal_filter(self, frame, prev_frame=None, alpha=0.5):
        if prev_frame is None : 
            return frame
        else : 
            result = cv2.addWeighted(frame, alpha, prev_frame, 1-alpha, 0)
            return result

class CameraData:
    def __init__(self, 
                 data_dir=DATA_DIR):
        self.results = []
        self.data_dir = data_dir
        self.data_template = {
            'color':
                {'raw':None,
                 'annot':None},
            'depth':
                {'raw':None,
                 'image':None},
            'gripper_loc':None,
            'items_loc':{},
            'config':None
        }
        
        self._setup_dir()
    
    def _setup_dir(self):
        t = time.localtime()
        self.current_time = f'waktu-{time.strftime("%Y-%m-%d %H-%M-%S", t)}'
        self.data_directory = os.path.join(self.data_dir, self.current_time)
        print(f'Saving data to the {self.data_directory}')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(os.path.join(self.data_directory, 'depth'))
        os.makedirs(os.path.join(self.data_directory, 'color'))
        os.makedirs(os.path.join(self.data_directory, 'data'))
    
    def set_data(self):
        self.i = 0
        self.cur_data = self.data_template
        return self.cur_data
    
    def append(self, data):
        self.results.append(data)
    
    def get_data(self):
        return self.results
    
    def save_current(self):
        
        with open(os.path.join(self.data_directory, 'data', f'data_{self.i}')+'.txt', 'w') as f:
            f.write(str(self.cur_data))
        
        color_image = self.cur_data['color']['annot']
        depth_image = self.cur_data['depth']['image']
        
        if color_image is not None:
            cv2.imwrite(os.path.join(self.data_directory, 'color', f'color_{self.i}')+'.jpg', color_image)
        if depth_image is not None:
            cv2.imwrite(os.path.join(self.data_directory, 'depth', f'depth_{self.i}')+'.jpg', depth_image)
        
        self.i += 1

    def save(self):
        pass
    
    def process(self):
        pass


if __name__ == '__main__':
    cam = DepthCamera(cam=0, thread_progress=False)
    cam.config(depth=False)
    cam.run()