# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:24:40 2015

@author: mick
"""
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")
import Tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import scipy.ndimage.interpolation as sip
import re
import tkFileDialog as tkf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time


class inter:
    def __init__(self):
        self.seed = 1
        root = tk.Tk()
        window = tk.LabelFrame(root,text="Test")
        window.pack()
        button = tk.Button(window,text="Beenden",command=root.destroy)
        button.pack()
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        self.canvas1 = FigureCanvasTkAgg(fig1,master=window)
        self.canvas1.get_tk_widget().pack()
        self.canvas2 = FigureCanvasTkAgg(fig2,master=window)
        self.canvas2.get_tk_widget().pack()
        button = tk.Button(window,text="Malen",command=self.draw)
        button.pack()
        plt.figure(2)
        root.mainloop()

    def draw(self):
        x = np.linspace(0,10,num=50)
        fig = plt.gcf()
        for _ in x:
            y = np.cos(self.seed * _)
            plt.plot(_,y,"rx")
            fig.canvas.draw()
        self.seed += 1

a = inter()
