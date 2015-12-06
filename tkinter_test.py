# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:57:27 2015
@author: mick
"""
matplotlib.use("TKAgg")
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Tkinter as tk
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

        self.input_fig = plt.figure("input",figsize=(5,5),dpi=108)
        self.input_canvas = FigureCanvasTkAgg(self.input_fig,master=input_frame)
        self.input_canvas.get_tk_widget().pack(side=tk.TOP)
        saveleft = tk.Button(input_frame,text="Bild speichern",command=self.save_left)
        saveleft.pack(side=tk.BOTTOM)
        output_frame = tk.LabelFrame(graph_frame,text="berechnetes Sinogramm/Rekonstruktion",padx=5,pady=5)
        output_frame.pack(fill=tk.BOTH,padx=10,pady=10,side=tk.RIGHT)
        self.output_fig = plt.figure("output",figsize=(5,5),dpi=108)
        self.output_canvas = FigureCanvasTkAgg(self.output_fig,master=output_frame)
        self.output_canvas.get_tk_widget().pack(side=tk.TOP)
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
#        shepp = tk.Radiobutton(filters,text="Shepp-Logan",variable=self.filterchoice,value=2,indicatoron=0,command=self.filtering)
#        shepp.pack(fill=tk.X)

        leave = tk.Button(sidebar,text="Beenden",command=root.destroy)
        leave.pack(fill=tk.X,side=tk.BOTTOM)

    def read_image(self):
        filename = tkf.askopenfilename()
        if re.search(r"(\.npy$)",filename):
            self.input_array = np.load(filename)
        else:
            self.input_array = np.loadtxt(filename)

        self.res = self.input_array.shape[0]
        self.wres = int(self.res*1.4)
        self.working_array = np.zeros((self.wres,self.wres))
        self.working_array[self.res*0.2:1.2*self.res,self.res*0.2:1.2*self.res] = self.input_array
        self.show_image(self.input_array,"input")

    def read_sino(self):
        filename = tkf.askopenfilename()
        arr = np.load(filename)
        self.filterchoice.set(0)
        self.sinogram = arr
        self.angles = arr[:,-1]
        self.wres = self.sinogram.shape[1] - 1
        self.res = self.sinogram[-1,0]
        self.show_image(self.sinogram,"input")

    def save_left(self):
        filename = tkf.asksaveasfilename()
        np.save(filename,self.input_array)

    def save_right(self):
        filename = tkf.asksaveasfilename()
        np.save(filename,self.output_array)

    def show_image(self,data,target):
        plt.figure(target)
        plt.imshow(data,aspect="auto")
        plt.set_cmap("gray")
        plt.axis("off")
        plt.xticks([]), plt.yticks([])
        fig = plt.gcf()
        fig.tight_layout(pad=0,w_pad=0,h_pad=0)
        fig.canvas.draw()

    def get_indices(self, angle):
        """
        Erstellt eine 2xMxN Matrix mit den Indizes des Originalbilds, aus denen
        der entsprechende Pixel im gedrehten Bild zusammengesetzt ist.
        """
        angle *= 2*np.pi/360.
        return np.fromfunction(lambda x,y,z: (z - (self.wres-1)/2)*
            np.sin(angle + x*np.pi/2) + (-1)**x * (y - (self.wres-1)/2)*
            np.cos(angle - x*np.pi/2) + self.wres/2, (2,self.wres,self.wres))

    def rotate_image(self, angle,image):
        smallest = np.min(image)
        rotated_image = sip.map_coordinates(image,self.get_indices(angle))
        rotated_image[rotated_image < smallest] = smallest
        return rotated_image

    def create_sinogram(self):
        self.angles = np.linspace(0,self.arc.get(),self.angle_picker.get())
        self.sinogram = np.zeros((self.angle_picker.get(),self.wres+1))
        for angle in range(self.angle_picker.get()):
            self.sinogram[angle,:-1] = np.sum(
                self.rotate_image(self.angles[angle],self.working_array),axis=1)
        self.sinogram[:,-1] = self.angles
        self.sinogram[-1,0] = self.res
        self.output_array = self.sinogram
        self.show_image(self.sinogram,"output")

    def back_projection(self):
        self.filterchoice.set(0)
        self.input_array = self.sinogram
        self.show_image(self.input_array,"input")
        image = np.zeros((self.wres,self.wres))
        for line in self.sinogram:
            image += self.rotate_image(-line[-1]-90,
               np.ones((self.wres,self.wres)) * line[:-1])
        image = image[self.res*0.2:1.2*self.res,self.res*0.2:1.2*self.res]
        image = image/np.max(image) * 255
        self.output_array = image
        self.ubp = image
        self.show_image(self.ubp,"output")

    def filtering(self):
        if self.filterchoice.get() == 0:
            self.output_array = self.ubp
        elif self.filterchoice.get() == 1:
            self.output_array = self.ramp_filter()
        elif self.filterchoice.get() == 2:
            self.output_array = self.shepp_logan()

        self.show_image(self.output_array,"output")


    def ramp_filter(self):
        image = self.ubp
        ft_image = np.fft.fftshift(np.fft.fft2(image))
        ramp = np.fromfunction(lambda x,y: np.sqrt((x-self.res/2)**2 +
            (y - self.res/2)**2),ft_image.shape)
        ft_image *= ramp
        ft_image = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_image)))
        mask = np.ones(ft_image.shape,dtype=bool)
        mask[5:-5,5:-5] = False
        ft_image[mask] = np.min(ft_image)
        return ft_image

root = tk.Tk()
root.title("Computed Pytography")
test = ct(root)
root.mainloop()