from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import cv2
from utils import _temporal_filter

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