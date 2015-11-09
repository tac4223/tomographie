# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:03:23 2015

@author: sengery
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sip
import re


class ct(object):

    def __init__(self):
        self.filename = None
        self.input_img = None
        self.res = None
        self.sinogram = None

    def read_image(self, filename):
        if re.search(r"(\.npy$)",filename):
            self.input_img = np.load(filename)

        else:
            self.input_img = np.loadtxt(filename)

        self.res = self.input_img.shape[0]
        self.filename = filename
        self.show_image(self.input_img)

    def input_chain(self,prompt,expected_type):
        """
        prompt: String der als Prompt für den Nutzer angezeigt werden soll.
        expected_type: Welcher Variablentyp als Eingabe akzeptiert wird.
        Stellt standardisierte Möglichkeit für Nutzereingaben zur Verfügung.
        """
        query = raw_input(prompt)
        try:
            return expected_type(query)
        except:
            print("{0} ist keine gültige Eingabe, bitte wiederholen".\
                format(query))
            self.input_chain(prompt, expected_type)

    def show_image(self,image):
        current_image = plt.imshow(image)
        current_image.set_cmap("gray")
        plt.show(current_image)

    def get_angles(self):

        self.arc = self.input_chain("\nProjektion ueber wieviel Grad?\n(180/360): ",int)
        if self.arc in [180,360]:
            pass
        else:
            print("Ungültige Eingabe, bitte 180 oder 360 auswählen.")
            self.get_angles()

        self.angle_count = self.input_chain("\nWieviele Punkte sollen auf "\
            "Projektionsbogen gewählt werden?\nGanze Zahl: ", int)
        self.angles = np.linspace(0,self.arc,self.angle_count)

    def get_indices(self, angle):
        """
        Erstellt eine 2xMxN Matrix mit den Indizes des Originalbilds, aus denen
        der entsprechende Pixel im Sinogramm zusammengesetzt ist.
        """
        angle *= 2*np.pi/360.
        return np.fromfunction(lambda x,y,z: (z - (self.res-1)/2)*
            np.sin(angle + x*np.pi/2) + (-1)**x * (y - (self.res-1)/2)*
            np.cos(angle - x*np.pi/2) + self.res/2, (2,self.res,self.res))

    def rotate_image(self, angle=0, image=None):
        if image == None: image = self.input_img
        rotated_image = sip.map_coordinates(image,self.get_indices(angle))
        rotated_image[rotated_image < 0] = 0
        return rotated_image

    def create_sinogram(self):
        self.sinogram = np.zeros((self.angle_count,self.res+1))
        for angle in range(self.angle_count):
            self.sinogram[angle,:-1] = np.sum(
                self.rotate_image(self.angles[angle]),axis=1)
            self.sinogram[angle,-1] = self.angles[angle]
        self.show_image(self.sinogram)


a = ct()
a.read_image("bilder\CT512.npy")
a.get_angles()
a.create_sinogram()