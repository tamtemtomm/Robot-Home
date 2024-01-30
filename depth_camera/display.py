import customtkinter as ctk
import cv2
import threading
from PIL import Image
from config import *
from tools import DepthCamera

class Frame(ctk.CTkFrame):
    def __init__(self, container, text, mode='image', side='left'):
        ctk.CTkFrame.__init__(self, container)
        self.setup(text=text, side=side)
        
        self.config_size = {
            'image_size': [640, 480],
            'current_size': [640, 480],
            'min_size': [100, 100],
        }
    
    def setup(self, text, side):
        self.out = ctk.CTkLabel(self, text=text, font= ("sans-serif", 30))
        self.out.pack(fill='both', expand=True, padx=5, pady=5)
        self.out.configure(anchor='center')
        self.img_label = ctk.CTkLabel(self, text="")
        self.img_label.pack(side=side, fill="both", expand=True, padx=5, pady=5)
    
    def img_update(self, img):
        self.img_label.configure(image=img)
        self.img = img

class App():
    def __init__(self, title='Orbecc Camera GUI', size=None, icon=ICON_PATH):
        self.run = False
        
        self.window = ctk.CTk()
        self.window.iconbitmap(icon)
        self.window.title(title)
        
        self.camera = DepthCamera(cam=0)
        self.camera.config(depth=False)
        
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
        
    def __add_color_frame(self):
        self.color_frame_display = Frame(self.window, 'Color Frame')
        self.color_frame_display.place(relx=0, rely=0.4, x=0, anchor='w')
        
    def __add_depth_frame(self):
        self.depth_frame_display = Frame(self.window, 'Depth Frame')
        self.depth_frame_display.place(relx=1, rely=1, x=-10, anchor='e')
    
    def loop(self):
        depth_img, _, color_img = self.camera.get_frame()
        
        if self.camera.color:
            self.__add_color_frame()
            color_img =  self._convert_to_pil(color_img)
            self.color_frame_display.img_update(color_img)
            
        if self.camera.depth : 
            self.__add_depth_frame()
            depth_img= self._convert_to_pil(depth_img, depth=True)
            self.depth_frame_display.img_update(depth_img)
        
        if self.run:
            self.window.after(1, self.loop)
    
    def start(self):
        self.run = True
        if self.camera.color:
            self.__add_color_frame()
            
        if self.camera.depth : 
            self.__add_depth_frame()
            
        self.loop()
        self.window.mainloop()
    
    def loop(self):
        depth_img, _, color_img = self.camera.get_frame()
        if self.camera.color:
            color_img =  self._convert_to_pil(color_img)
            self.color_frame_display.img_update(color_img)
        
        if self.camera.depth:
            depth_img =  self._convert_to_pil(depth_img)
            self.depth_frame_display.img_update(depth_img)
        
        if self.run:
            self.window.after(1, self.loop)

    def close(self):
        self.run = False
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
    app = App()
    app.start()