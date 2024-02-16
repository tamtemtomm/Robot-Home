from depth_camera.config import *
import os, time, pickle
import numpy as np
from depth_camera.utils import _euclidian_distance, _pixel_to_distance

class CameraData:
    def __init__(self, 
                 data_dir=DATA_DIR,
                 convert_to_distance=True,
                 bracket_theta = 30):
        self.convert_to_distance = convert_to_distance
        self.bracket_theta = bracket_theta
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
            # 'orientation'   : self.get_barcode_orientation(),
        }
        return self.results
    
    def get_min_distance(self):
        min_distance = 9999999
        min_location = None
        min_target = None
        result = {
            'grip_location'     : None,
            'item_location'     : None,
            'distance'          : None,
            'target'            : None,
        }
        try : 
            grip_loc = self.cur_data['gripper_loc']['location']
        except :
            grip_loc = [0, 0, 0]
        
        for item in self.cur_data['items_loc']:
            if item:
                if grip_loc is not None:
                    for i in self.cur_data['items_loc'][item]:
                        distance = _euclidian_distance(grip_loc, i['location'])
                        if distance < min_distance:
                            min_distance = _pixel_to_distance(distance) if self.convert_to_distance else distance
                            min_location = i['location']
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