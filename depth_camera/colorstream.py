from config import *
from utils import _temporal_filter, _add_border, _add_square

import cv2
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator
from pyzbar import pyzbar
# from inference_sdk import InferenceHTTPClient
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ColorStream:
    def __init__(self, cam,
                 barcode=True,
                 ):
        
        self.cap = cv2.VideoCapture(cam)
        self.barcode_auth = 'hahahaha' if barcode else None 
        
        # try : 
        #     self.CLIENT = InferenceHTTPClient(
        #         api_url="https://detect.roboflow.com",
        #         api_key="1UnUQCCfSuu44HS6CrHe"
        #     )
        
        # except : 
        #     self.CLIENT = None
            
    def get_frame(self, 
                  img_depth=None, 
                  model=None, 
                  gripper_model=None,
                  barcode_model=None,
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
                self.barcode_model = barcode_model
                color_image = self._annotate_barcode_segment(color_image, color_image_raw, img_depth)
            
            if data : 
                self.data['color']['annot'] = color_image
                return color_image, self.data
            
        else : 
            return None, None
    
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
    
    def _yolo_barcode(self, img, img_raw, img_depth):
        results = self.gripper_model.predict(img_raw, verbose=False)
        if results:
            for r in results:
                print(r.boxes)
                if r.boxes:
                    for box in r.boxes:
                        img = self._annotate_barcode_segment2(img, box, img_depth)
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
    
    def _annotate_gripper_segment(self, img, box, img_depth=None):
        bbox = box.xyxy[0]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
        depth_estimation = None
        
        location = (center[0], center[1], depth_estimation)
        img = _add_square(img, bbox, center, location, 'GRIPPER')
        self.data['gripper_loc'] = {'bbox':bbox,
                                    'location':location}
        return img

    def _annotate_barcode_segment(self, img, img_raw, img_depth=None):
        depth_estimation = None
    
        for barcode in pyzbar.decode(img_raw):
            if barcode :
                barcode_data = barcode.data.decode('utf-8')
                if barcode_data is not None:
                    x1, y1, w, h  = barcode.rect
                    x2, y2 = x1 + w, y1 + h
                    
                    box = (x1, y1, x2, y2)
                    bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
                    location = (center[0], center[1], depth_estimation)
                    
                    img = _add_square(img, box, center, location, 'BARCODE')
                    
                    self.data['barcode_loc'] = {'bbox':bbox,
                                                'location':location,
                                                'data': barcode_data}
                    
                    break
                else :
                    continue
        
        return img
    
    def _annotate_barcode_segment2(self, img, box, img_depth=None):
        bbox = box.xyxy[0]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
        depth_estimation = None
        
        location = (center[0], center[1], depth_estimation)
        img = _add_square(img, bbox, center, location, 'BARCODE')
        self.data['barcode_loc'] = {'bbox':bbox,
                                    'location':location}
        return img