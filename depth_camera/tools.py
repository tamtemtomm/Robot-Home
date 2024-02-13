import cv2
import numpy as np
import torch
from primesense import openni2
from primesense import _openni2 as c_api
import os, shutil, time, json, pickle
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from pyzbar import pyzbar
import threading

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from config import *
from utils import _pixel_to_distance, _temporal_filter, _add_square, _add_border, euclidian_distance, _to_bbox

class DepthCamera :
    def __init__(self, 
                 cam = 1,
                 redist = REDIST_PATH,
                 data_dir = DATA_DIR,
                 thread_progress=True,
                 height=200):
        
        self.cam = cam
        self.redist = redist
        self.data_dir = data_dir
        self.device = device
        self.focal_length = 60
        self.height = height
        
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
        
        # self.depth = False
        
        self.data = CameraData()
    
    def config(self, 
                yolo=True,
                temporal_filter=False,
                colormap=False,
                save_data = False):
        
        self.width = 640
        self.height = 480
        self.fps = 30
        
        if self._check_params(
            yolo,
            temporal_filter,
            colormap,
            save_data) :
            return 
        
        ##---------------------------------------------------------------------------------------------------------
        # YOLO MODEL INITIALISATION
        
        self.model, self.gripper_model = self._model()
        
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
        
    def run(self, verbose=False):
        
        if self.thread_progress:
            self.thread.start()
        
        self.loop(verbose=verbose)
            
        self.close()
        
    ##-----------------------------------------------------------------------------------------------
    ## Add Extra Functions
    
    def close(self):
        # print(f'Result : {self.data.get_data()}')

        if self.depth:
            self.depth_stream.close()
        self.color_stream.close()
    
    def loop(self, show=True, verbose=False):
        if self.save_data:
            self.data.setup()
        
        while True :
            self.depth_image, self.img_depth, self.color_image = self.get_frame(show=show, verbose=verbose)
            
            if self.cur_data['gripper_loc']:
                print(self.data.cur_process())
            
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
                                                            temporal_filter=self.temporal_filter,
                                                            data = self.cur_data)
        if self.color_image is not None:
            if show: 
                cv2.imshow("Color Image", self.color_image)

        if verbose:
            print(self.cur_data)

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
        
        # gripper_model = None

        return model, gripper_model

    def _check_params( self,
            yolo,
            temporal_filter,
            colormap,
            save_data):
        
        # for args in [depth, color, yolo, colormap, temporal_filter, save_data]:
        #     if isinstance(args, bool) == False:
        #         print(f'(depth, color, yolo, colormap, temporal_filter, save_data) parameter only accept boolean datatype')
        #         return True
        # for args in [width, height, fps]:
        #     if isinstance(args, int) == False:
        #         print(f'(width, height, fps) parameter only accept int datatype')
        #         return True
            
        self.yolo = yolo
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
            depth_image = _temporal_filter(depth_image, self.prev_depth_image)
            self.prev_depth_image = depth_image
           
        if data:
            self.data = data
            self.data['depth']['raw'] = img_depth
            self.data['depth']['image'] = depth_image

            return depth_image, img_depth, self.data
            
        return depth_image, img_depth, None
    
    def close(self):
        openni2.unload()

class ColorStream:
    def __init__(self, cam,
                 barcode=True,
                 ):
        self.cap = cv2.VideoCapture(cam)
        self.barcode_auth = 'hahahaha' if barcode else None 
        
    def get_frame(self, 
                  img_depth=None, 
                  model=None, 
                  gripper_model=None,
                  temporal_filter=False,
                  data = None
                ):
        
        if data :
            self.data = data

        self.temporal_filter = temporal_filter
        if self.temporal_filter:
            self.prev_color_image = None
        
        _, color_image_raw = self.cap.read()

        if color_image_raw is not None:
        
            if data:
                self.data['color']['raw'] = color_image_raw
                
            if self.temporal_filter :
                color_image_raw= _temporal_filter(color_image_raw, self.prev_color_image)
                self.prev_color_image = color_image_raw
            
            if model is not None:
                self.model = model
                color_image = self._yolo(color_image_raw, img_depth)
            
            if gripper_model is not None:
                self.gripper_model = gripper_model
                color_image = self._yolo_gripper(color_image, color_image_raw, img_depth)    
            
            if self.barcode_auth is not None:
                color_image = self._annotate_barcode_segment(color_image, color_image_raw, img_depth)
            
            if data : 
                self.data['color']['annot'] = color_image
                return color_image, self.data
            
        else : color_image = None
            
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
    
    def _yolo_gripper(self, img, img_raw, img_depth):
        results = self.gripper_model.predict(img_raw, verbose=False)
        if results:
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        img = self._annotate_gripper_segment(img, box, img_depth)
        return img
    
    def _annotate_segment(self, img, box, mask, annotator, img_depth) : 
        bbox = box.xyxy[0]
        class_name = self.model.names[int(box.cls)]
        border = mask.xy[0]
        
        annotator.box_label(bbox, class_name)
        img = _add_border(img, border)
        
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
        self.data['items_loc'][class_name].append({'bbox':bbox.numpy(), 
                                                   'location':location, 
                                                   'mask':mask_segment})
        
        return img, annotator
    
    def _annotate_gripper_segment(self, img, box, img_depth):
        bbox = box.xyxy[0]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
        depth_estimation = None
        
        location = (center[0], center[1], depth_estimation)
        img = _add_square(img, bbox, center, location)
        self.data['gripper_loc'] = {'bbox':bbox,
                                    'location':location}
        return img

    def _annotate_barcode_segment(self, img, img_raw, img_depth=None):
        result_image = None
        depth_estimation = None
        
        for barcode in pyzbar.decode(img_raw):
            if barcode :
                barcode_data = barcode.data.decode('utf-8')
                if barcode_data == self.barcode_auth:
                    x1, y1, w, h  = barcode.rect
                    x2, y2 = x1 + w, y1 + h
                    
                    box = (x1, y1, x2, y2)
                    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
                    location = (center[0], center[1], depth_estimation)
                    
                    img = _add_square(img, box, center, location)
                    
                    self.data['barcode_loc'] = {'bbox':bbox,
                                                'location':location,
                                                'data': barcode_data}
                    
                    break
                else :
                    continue
        
        return img
        
class CameraData:
    def __init__(self, 
                 data_dir=DATA_DIR,
                 convert_to_distance=True):
        self.convert_to_distance = convert_to_distance
        self.data = {
            'data':[],
            'config':None
        }
        self.data_dir = data_dir
        self.data_template = {
            'color':
                {'raw':None,
                 'annot':None},
            'depth':
                {'raw':None,
                 'image':None},
            'gripper_loc':None,
            'barcode_loc':None,
            'items_loc':{},
        }
        
        self.i = 0
    
    def setup(self):
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
        self.cur_data = self.data_template
        self.results = None
        return self.cur_data
    
    def append(self, data, save=False):
        self.cur_data = data
        self.data['data'].append(data)
        
        if save :
            self.save_cur(data)
    
    def save_cur(self, data):
        with open(os.path.join(self.data_directory, 'data', f'data_{self.i}.txt'), 'wb') as f:
            pickle.dump(data, f)
        
        color_image = data['color']['annot']
        depth_image = data['depth']['image']
        
        if color_image is not None:
            cv2.imwrite(os.path.join(self.data_directory, 'color', f'color_{self.i}')+'.jpg', color_image)
        if depth_image is not None:
            cv2.imwrite(os.path.join(self.data_directory, 'depth', f'depth_{self.i}')+'.jpg', depth_image)
    
        self.i = self.i + 1

    def save(self):
        with open(os.path.join(self.data_directory, 'data'+'.txt'), 'wb') as f:
            pickle.dump(self.data, f)
    
    def cur_process(self):
        self.results = {
            'min'           : self.get_min_distance(),
            'orientation'   : self.get_barcode_orientation(),
        }
        return self.results
    
    def get_min_distance(self):
        min_distance = 9999999
        min_location = None
        min_target = None
        result = None
        
        grip_loc = self.cur_data['gripper_loc']['location']
        
        if grip_loc is not None:
            for item in self.cur_data['items_loc']:
                if item:
                    for i in self.cur_data['items_loc'][item]:
                        distance = euclidian_distance(grip_loc[:2], i['location'][:2])
                        if distance < min_distance:
                            min_distance = _pixel_to_distance(distance) if self.convert_to_distance else distance
                            min_location = i['location'][:2]
                            min_target = (grip_loc[0] - i['location'][0], grip_loc[1] - i['location'][1])
        
                        result = {
                            'grip_location'     : grip_loc,
                            'item_location'     : min_location,
                            'distance'          : min_distance,
                            'target'            : min_target
                        }
        
        return result
    
    def get_barcode_orientation(self):
        orientation = None
        
        if self.cur_data['gripper_loc'] is not None and self.cur_data['barcode_loc'] is not None:
            grip_loc = self.cur_data['gripper_loc']['location']
            barcode_loc = self.cur_data['barcode_loc']['location']
            
            y, x = np.abs(grip_loc[1] - barcode_loc[1])/np.abs(grip_loc[0] - barcode_loc[0])
            orientation = np.arctan(y/x)*180/np.pi
            orientation = orientation if (barcode_loc[0] - grip_loc[0] >= 0) else (180 - orientation)

        return orientation
                        
if __name__ == '__main__':
    cam = DepthCamera(cam=0,)
    cam.config()
    cam.run(verbose=False)