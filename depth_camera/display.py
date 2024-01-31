import customtkinter as ctk
import cv2
import threading
from PIL import Image
from config import *
from tools import DepthCamera

class Frame(ctk.CTkFrame):
    def __init__(self, container, text, side='left'):
        ctk.CTkFrame.__init__(self, container)
        self.setup(text=text, side=side)
        
        self.config_size = {
            'image_size': [640, 480],
            'current_size': [640, 480],
            'min_size': [100, 100],
        }
    
    def setup(self, text, side):
        self.out = ctk.CTkLabel(self, text=text, font= ("sans-serif", 15))
        self.out.pack(fill='both', expand=True, padx=5, pady=5)
        self.out.configure(anchor='center')
        self.img_label = ctk.CTkLabel(self, text="")
        self.img_label.pack(side=side, fill="both", expand=True, padx=5, pady=5)
    
    def img_update(self, img):
        self.img_label.configure(image=img)
        self.img = img

class Button(ctk.CTkButton):
    def __init__(self, container,  **kwargs):
        ctk.CTkButton.__init__(self, container, **kwargs)
        self._config()
        
    def _config(self, text):
        self.out = ctk.CTkLabel(self, text=text, font= ("sans-serif", 15))

class CheckBox(ctk.CTkCheckBox):
    def __init__(self, container, **kwargs):
        ctk.CTkCheckBox.__init__(self, container, **kwargs)
        self._config()
    
    def _config(self):
        pass
        

class App():
    def __init__(self, 
                 camera,
                 title='Orbecc Camera GUI', 
                 size=None, 
                 icon=ICON_PATH):
        self.isrun = False
        
        self.window = ctk.CTk()
        self.window.iconbitmap(icon)
        self.window.title(title)
        
        self.camera = camera
        
        self.config_size = {
            'init_size': [1360, 1080],
            'current_size': [1360, 1080],
            'min_size': [920, 720],
            'im_size':[self.camera.width, self.camera.height]
        }
        
        self.window.minsize(self.config_size['min_size'][0], self.config_size['min_size'][1])        
        if size:
            self.window.geometry(f"{size}")
        else :
            self.window.geometry(f"{self.config_size['init_size'][0]}x{self.config_size['init_size'][1]}")
        
        if self.camera.thread_progress:
            self.camera.thread.start()
        
        self.__setup()
        
    def __setup(self):
        if self.camera.color:
            self.__add_color_frame()
            
        if self.camera.depth : 
            self.__add_depth_frame()
        
        self.__add_temporal_filter_cb()
        self.__add_yolo_cb()
        
    def __add_color_frame(self):
        self.color_frame_display = Frame(self.window, 'Color Frame')
        self.color_frame_display.place(relx=0, rely=0.4, x=0, anchor='w')
        
    def __add_depth_frame(self):
        self.depth_frame_display = Frame(self.window, 'Depth Frame')
        self.depth_frame_display.place(relx=1, rely=0.4, x=-10, anchor='e')
    
    def __add_temporal_filter_cb(self):
        temporal_filter_cb = CheckBox(self.window, 
                                    #   variable=self.camera.temporal_filter
                                      )
        temporal_filter_cb.place(x=10, y=7)
    
    def __add_yolo_cb(self):
        self.yolo_var = ctk.StringVar(value='on')
        yolo_cb = CheckBox(self.window, 
                        #    variable=self.yolo_var, command=self.__cb_command('yolo'),
                           onvalue='on', offvalue='off', text='YOLOv8')
        yolo_cb.place(x=180, y=7)
        
    def __cb_command(self, arg):
        if arg == 'temporal':
            pass
        if arg == 'yolo':
            # if self.yolo_var == 'on':
            #     self.camera.yolo = True
            # else : self.camera.yolo = False
            
            pass
            
    def run(self):
        self.isrun = True
        self.loop()
        self.window.mainloop()
    
    def loop(self):
        depth_img, _, color_img = self.camera.get_frame(show=False)
        if self.camera.color:
            color_img =  self._convert_to_pil(color_img)
            self.color_frame_display.img_update(color_img)
        
        if self.camera.depth:
            depth_img =  self._convert_to_pil(depth_img)
            self.depth_frame_display.img_update(depth_img)
        
        if self.isrun:
            self.window.after(14, self.loop)

    def close(self):
        self.isrun = False
        self.camera.close()
        self.window.destroy()
    
    def _convert_to_pil(self, img, depth=False):
        if depth:
            img = Image.fromarray(img)
            img = ctk.CTkImage(img, size=(self.config_size['im_size'][0], self.config_size['im_size'][1]))
            
        else : 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ctk.CTkImage(img, size=(self.config_size['im_size'][0], self.config_size['im_size'][1]))
        
        return img

if __name__ == '__main__':
    camera = DepthCamera(cam=0, thread_progress=True)
    app = App(camera)
    app.run()