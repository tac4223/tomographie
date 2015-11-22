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
        self.var_init()

        sidebar = tk.Frame(master,padx=5,pady=5)
        sidebar.pack(side=tk.LEFT)

        graph_frame = tk.Frame(master,padx=5,pady=5)
        graph_frame.pack(fill=tk.BOTH,side=tk.RIGHT)

        input_frame = tk.LabelFrame(graph_frame,text="Ausgangsbild/geladenes Sinogramm",padx=5,pady=5)
        input_frame.pack(fill=tk.BOTH,padx=10,pady=10,side=tk.LEFT)
        self.input_canvas = tk.Canvas(input_frame,width=512,height=512)
        self.input_canvas.pack(side=tk.LEFT)
        output_frame = tk.LabelFrame(graph_frame,text="berechnetes Sinogramm/Rekonstruktion",padx=5,pady=5)
        output_frame.pack(fill=tk.BOTH,padx=10,pady=10,side=tk.RIGHT)
        self.output_canvas = tk.Canvas(output_frame,width=512,height=512)
        self.output_canvas.pack(fill=tk.BOTH,side=tk.RIGHT)



        self.sino_settings = tk.LabelFrame(sidebar, text="Sinogramm",padx=5,pady=5)
        self.arc = tk.IntVar(self.sino_settings)
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
        save_sino = tk.Button(self.sino_settings,text="Sinogramm\nspeichern",command=self.save_image)
        save_sino.pack(fill=tk.X)


        filter_settings = tk.LabelFrame(sidebar, text="Rekonstruktion",padx=5,pady=5)
        filter_settings.pack(fill=tk.X,side=tk.TOP)
        note = tk.LabelFrame(filter_settings,text="optional:",padx=5,pady=5)
        note.pack()
        loadsino = tk.Button(note,text="Sinogramm laden",command=self.read_sino)
        loadsino.pack(fill=tk.X)
        filterchoice1 = tk.Radiobutton(filter_settings,text="Ungefiltert", variable=self.filter,value=0)
        filterchoice1.pack()
        filterchoice2 = tk.Radiobutton(filter_settings,text="Ramp-Filter", variable=self.filter,value=1)
        filterchoice2.pack()
        go_reco = tk.Button(filter_settings,text="Starten",command=self.unfiltered_back)
        go_reco.pack(fill=tk.X)
        save_reco = tk.Button(filter_settings,text="Rekonstruktion\nspeichern",command=self.save_image)
        save_reco.pack(fill=tk.X)



    def var_init(self):
        """
        Um sicher zu gehen dass auch beim Neuladen der Bilder jeweils alle
        Variablen im Ausgangszustand sind, bietet sich diese Funktion an.
        """

        self.filename = None
        self.input_array = None
        self.output_array = None
        self.res = None
        self.sinogram = None
        self.ubp = None
        self.fbp = None
        self.do_recon = 0
        self.filter = 0
        self.ltemp = None
        self.rtemp = None


    def read_image(self):
        self.var_init()
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
        self.input_array = arr
        self.angles = arr[:,-1]
        self.res = self.input_array.shape[1] - 1
        self.show_image(self.input_array,0,self.input_canvas)


    def save_image(self):
        filename = tkf.asksaveasfilename()
        np.save(filename,self.output_array)

    def show_image(self,data,temp,target):
        data *= 1./np.max(data) * 256
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
        self.angle_count = self.angle_picker.get()
        self.angles = np.linspace(0,self.arc.get(),self.angle_count)
        self.sinogram = np.zeros((self.angle_count,self.res+1))

        for angle in range(self.angle_count):
            self.sinogram[angle,:-1] = np.sum(
                self.rotate_image(self.angles[angle],self.input_array),axis=1)
        self.sinogram[:,-1] = self.angles
        self.output_array = self.sinogram
        self.show_image(self.sinogram,1,self.output_canvas)
        self.do_recon = 1

    def unfiltered_back(self):
        if self.do_recon == 1:
            self.show_image(self.output_array,0,self.input_canvas)
            self.input_array = self.output_array
        image = np.zeros((self.res,self.res))
        for line in self.input_array:
            image += self.rotate_image(-line[-1]-90,
            np.ones((self.res,self.res)) * line[0:-1])
        self.output_array = image
        if self.filter == 1:
            self.ramp_filter(image)
        elif self.filter == 0:
            self.show_image(self.output_array,1,self.output_canvas)

    def ramp_filter(self,image):
        ft_image = np.fft.fftshift(np.fft.fft2(image))
        ramp = np.fromfunction(lambda x,y: np.sqrt((x-self.res/2)**2 +
            (y - self.res/2)**2),ft_image.shape)
        edge = ramp >= self.res/2.1
        ft_image *= ramp
        ft_image = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_image)))
        ft_image[edge] = 0
        self.output_array = ft_image

root = tk.Tk()
root.title("Computed Pytography")
test = ct(root)
root.mainloop()