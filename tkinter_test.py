# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:57:27 2015
@author: mick
"""
import matplotlib
matplotlib.use("TKAgg")
import Tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import scipy.ndimage.interpolation as sip
import re
import tkFileDialog as tkf

class ct:

    def __init__(self, master):
        sidebar = tk.Frame(master,padx=5,pady=5)
        sidebar.pack(side=tk.LEFT)

        graph_frame = tk.Frame(master,padx=5,pady=5)
        graph_frame.pack(fill=tk.BOTH,side=tk.RIGHT)

        input_frame = tk.LabelFrame(graph_frame,text="Ausgangsbild/geladenes Sinogramm",padx=5,pady=5)
        input_frame.pack(fill=tk.BOTH,padx=10,pady=10,side=tk.LEFT)
        self.input_canvas = tk.Canvas(input_frame,width=512,height=512)
        self.input_canvas.pack(side=tk.TOP)
        saveleft = tk.Button(input_frame,text="Bild speichern",command=self.save_left)
        saveleft.pack(side=tk.BOTTOM)
        output_frame = tk.LabelFrame(graph_frame,text="berechnetes Sinogramm/Rekonstruktion",padx=5,pady=5)
        output_frame.pack(fill=tk.BOTH,padx=10,pady=10,side=tk.RIGHT)
        self.output_canvas = tk.Canvas(output_frame,width=512,height=512)
        self.output_canvas.pack(fill=tk.BOTH,side=tk.TOP)
        saveright = tk.Button(output_frame,text="Bild speichern",command=self.save_right)
        saveright.pack(side=tk.BOTTOM)



        self.sino_settings = tk.LabelFrame(sidebar, text="Sinogramm",padx=5,pady=5)
        self.arc = tk.IntVar(self.sino_settings,180)
        self.sino_settings.pack(fill=tk.X,side=tk.TOP)
        loadimg = tk.Button(self.sino_settings,text="Ausgangsbild laden",command=self.read_image)
        loadimg.pack(fill=tk.X)
        self.arcchoice1 = tk.Radiobutton(self.sino_settings,text="Halbkreis", variable=self.arc,value=180)
        self.arcchoice1.pack()
        self.arcchoice2 = tk.Radiobutton(self.sino_settings,text="Vollkreis", variable=self.arc,value=360)
        self.arcchoice2.pack()
        tk.Label(self.sino_settings,text="Anzahl Scanwinkel").pack()
        self.angle_picker = tk.Scale(self.sino_settings,from_=0,to=512,orient=tk.HORIZONTAL)
        self.angle_picker.pack(fill=tk.X)
        go_sino = tk.Button(self.sino_settings,text="Starten",command=self.create_sinogram)
        go_sino.pack(fill=tk.X)


        reco = tk.LabelFrame(sidebar, text="Rekonstruktion",padx=5,pady=5)
        reco.pack(fill=tk.X,side=tk.TOP)
        note = tk.LabelFrame(reco,text="optional:",padx=5,pady=5)
        note.pack()
        loadsino = tk.Button(note,text="Sinogramm laden",command=self.read_sino)
        loadsino.pack(fill=tk.X)
        go_reco = tk.Button(reco,text="Starten",command=self.back_projection)
        go_reco.pack(fill=tk.X)

        filters = tk.LabelFrame(sidebar,text="Filter für Rekonstrukion",padx=5,pady=5)
        filters.pack(fill=tk.X,side=tk.TOP)
        self.filterchoice = tk.IntVar(filters,value=0)
        unfiltered = tk.Radiobutton(filters,text="Kein Filter",variable=self.filterchoice,value=0,indicatoron=0,command=self.filtering)
        unfiltered.pack(fill=tk.X)
        ramp = tk.Radiobutton(filters,text="Ramp-Filter",variable=self.filterchoice,value=1,indicatoron=0,command=self.filtering)
        ramp.pack(fill=tk.X)
        shepp = tk.Radiobutton(filters,text="Shepp-Logan",variable=self.filterchoice,value=2,indicatoron=0,command=self.filtering)
        shepp.pack(fill=tk.X)

    def read_image(self):
        filename = tkf.askopenfilename()
        if re.search(r"(\.npy$)",filename):
            self.input_array = np.load(filename)
        else:
            self.input_array = np.loadtxt(filename)

        self.res = self.input_array.shape[0]
        self.show_image(self.input_array,0,self.input_canvas)

    def read_sino(self):
        filename = tkf.askopenfilename()
        arr = np.load(filename)
        self.filterchoice.set(0)
        self.sinogram = arr
        self.angles = arr[:,-1]
        self.res = self.sinogram.shape[1] - 1
        self.show_image(self.sinogram,0,self.input_canvas)

    def save_left(self):
        filename = tkf.asksaveasfilename()
        np.save(filename,self.input_array)

    def save_right(self):
        filename = tkf.asksaveasfilename()
        np.save(filename,self.output_array)

    def show_image(self,data,temp,target):
        data = 1./np.max(data) * 256 * data
        size = 512,512
        if temp == 0:
            self.ltemp = Image.fromarray(data)
            self.ltemp = ImageTk.PhotoImage(self.ltemp.resize(size,Image.BICUBIC))
            target.create_image(0,0,image=self.ltemp,anchor="nw")

        elif temp == 1:
            self.rtemp = Image.fromarray(data)
            self.rtemp = ImageTk.PhotoImage(self.rtemp.resize(size,Image.BICUBIC))
            target.create_image(0,0,image=self.rtemp,anchor="nw")


    def get_indices(self, angle):
        """
        Erstellt eine 2xMxN Matrix mit den Indizes des Originalbilds, aus denen
        der entsprechende Pixel im gedrehten Bild zusammengesetzt ist.
        """
        angle *= 2*np.pi/360.
        return np.fromfunction(lambda x,y,z: (z - (self.res-1)/2)*
            np.sin(angle + x*np.pi/2) + (-1)**x * (y - (self.res-1)/2)*
            np.cos(angle - x*np.pi/2) + self.res/2, (2,self.res,self.res))

    def rotate_image(self, angle,image):
        rotated_image = sip.map_coordinates(image,self.get_indices(angle))
        rotated_image[rotated_image < 0] = 0
        return rotated_image

    def create_sinogram(self):
        self.angles = np.linspace(0,self.arc.get(),self.angle_picker.get())
        self.sinogram = np.zeros((self.angle_picker.get(),self.res+1))

        for angle in range(self.angle_picker.get()):
            self.sinogram[angle,:-1] = np.sum(
                self.rotate_image(self.angles[angle],self.input_array),axis=1)
        self.sinogram[:,-1] = self.angles
        self.output_array = self.sinogram
        self.show_image(self.sinogram,1,self.output_canvas)

    def back_projection(self):
        self.filterchoice.set(0)
        self.input_array = self.sinogram
        self.show_image(self.input_array,0,self.input_canvas)
        image = np.zeros((self.res,self.res))
        for line in self.sinogram:
            image += self.rotate_image(-line[-1]-90,
               np.ones((self.res,self.res)) * line[:-1])
        self.output_array = image
        self.ubp = image
        self.show_image(self.ubp,1,self.output_canvas)

    def filtering(self):
        self.input_array = self.ubp
        self.show_image(self.ubp,0,self.input_canvas)

        if self.filterchoice.get() == 0:
            self.output_array = self.ubp
        elif self.filterchoice.get() == 1:
            self.output_array = self.ramp_filter()

        self.show_image(self.output_array,1,self.output_canvas)


    def ramp_filter(self):
        image = self.ubp
        ft_image = np.fft.fftshift(np.fft.fft2(image))
        ramp = np.fromfunction(lambda x,y: np.sqrt((x-self.res/2)**2 +
            (y - self.res/2)**2),ft_image.shape)
        edge = ramp >= self.res/2.1
        ft_image *= ramp
        ft_image = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_image)))
        ft_image[edge] = 0
        return ft_image

root = tk.Tk()
root.title("Computed Pytography")
test = ct(root)
root.mainloop()