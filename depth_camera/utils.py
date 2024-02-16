import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from depth_camera.config import *

def _temporal_filter(frame, prev_frame=None, alpha=0.5):
    if prev_frame is None : 
        return frame
    else : 
        result = cv2.addWeighted(frame, alpha, prev_frame, 1-alpha, 0)
        return result

def _pixel_to_distance(pixel, depth=DEPTH, focal_length=FOCAL_LENGTH):
    return pixel*depth/focal_length

def _distance_to_pixel(distance, depth, focal_length):
    return focal_length*distance/depth

def _convert_to_pil(img, length, width, depth=False):
    if depth:
        img = Image.fromarray(img)
        img = ctk.CTkImage(img, size=(length, width))
        
    else : 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ctk.CTkImage(img, size=(length, width))
    
    return img

def _to_bbox(bbox):
    box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    center = (int(bbox[0] + (bbox[2] - bbox[0])/2), int(bbox[1] + (bbox[3] - bbox[1])/2))
    
    return box, center

def _add_square(img, box, center=None, location=None, text = None):
    x1, y1, x2, y2 = box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), DEFAULT_COLOR, 2)
    if text is not None:
        img = cv2.putText(img, f'{text}', (x1, y1), DEFAULT_FONT, 0.4, (0, 0, 255), 1, DEFAULT_LINE)
    
    if location is not None and center is not None:
        img = cv2.putText(img, f'{location}', center, DEFAULT_FONT, 
                            0.4, (0, 0, 255), 1, DEFAULT_LINE)
    return img

def _add_border(img, border):
    for a, b in border:
        img[int(b)-1, int(a)-1, 0] = 0
        img[int(b)-1, int(a)-1, 1] = 0
        img[int(b)-1, int(a)-1, 2] = 255
    return img

def _add_line(img, point1, point2):
    point1, point2 = int(point1), int(point2)
    return cv2.line(img, point1, point2, DEFAULT_COLOR, 1)

def _euclidian_distance(arr_a, arr_b):
    sum = 0
    for a , b in zip(arr_a, arr_b):
        sum += np.power((a-b), 2) 
    return np.sqrt(sum)