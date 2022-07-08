# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:47:00 2020

@author: lancery
"""

import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import tkinter


class vecteurs(list):
    def __init__(self, L):
        list.__init__(self)
        for i in range(len(L)):
            self.append(L[i])

    def displayr(self):
        return("({};{};{})".format(self[0], self[1], self[2]))

    def displayp(self):
        print("({};{};{})".format(self[0], self[1], self[2]))

    def norme(self):
        norm = 0
        for i in range(len(self)):
            norm = norm + self[i]**2
        return norm**0.5

    def angles(self, other):
        # résultat en radians
        return(np.arccos(self.Pscal(other)/(self.norme()*other.norme())))

    def Pscal(self, other):
        Ps = 0
        for i in range(len(self)):
            Ps = Ps + self[i]*other[i]
        return(Ps)

    def __add__(self, other):

        for i in range(len(self)):
            self[i] = self[i] + other[i]
        return self

    def __sub__(self, other):
        for i in range(len(self)):
            self[i] = self[i] - other[i]
        return self

    def __mul__(self, k):
        for i in range(len(self)):
            self[i] = self[i]*k
        return self

    def generimage(self, other, other2):
        x = self[0]
        y = self[1]
        z = self[2]
        x1 = other[0]
        y1 = other[1]
        z1 = other[2]
        x2 = other2[0]
        y2 = other2[1]
        z2 = other2[2]
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.scatter(x, y, z, color="black")
        ax.scatter(x1, y1, z1, color="red")
        ax.scatter(x2, y2, z2, color="blue")
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        xt = 6371000 * np.outer(np.cos(u), np.sin(v))
        yt = 6371000 * np.outer(np.sin(u), np.sin(v))
        zt = 6371000 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(xt, yt, zt, 0.3, cmap="binary", alpha=1)


# vec d'entré cartesien
# class Point_Terre(vecteurs):
#    def __init__(self,L):
#        vecteurs.__init__(self,L)
#    def cartesien(self):
#        return(self.displayp())
#    def cylindrique(self):
#        a=vecteurs([self.__getitem__(0),self.__getitem__(1),0])
#        return(vecteurs([a.norme(),a.angles(other=vecteurs([0,1,0])),self.__getitem__(2)]))
#    def spherique(self):
#        a=vecteurs([0,self.__getitem__(1),self.__getitem__(2)])
#        b=vecteurs([self.__getitem__(0),self.__getitem__(1),0])
#        return(vecteurs([self.norme(),a.angles(other=vecteurs([0,0,1])),b.angles(other=vecteurs([0,1,0]))]))


class Point_Terre(vecteurs):
    def __init__(self, L):
        vecteurs.__init__(self, L)

    def cartesien(self):
        rT = 6371*10**3
        self2 = self.spherique()
        self1 = vecteurs([self2[0], self2[1]*(2*np.pi) /
                         360, self2[2]*(2*np.pi)/360])
        x = rT*np.cos(self1[2])*np.sin(self1[1])
        y = rT*np.sin(self1[2])*np.sin(self1[1])
        z = rT*np.cos(self1[1])
        return(vecteurs([x, y, z]))

    def cylindrique(self):
        n = self.cartesien()
        a = vecteurs([n[0], n[1], 0])
        teta = np.arctan(n[1]/n[0])
        return(vecteurs([a.norme(), teta, n[2]]))

    def spherique(self):
        return(vecteurs([self[0], 90-self[1], self[2]]))

    def distance(self, other):
        return self.angles(other)*rT

    def latencefibre(self, other):
        cfibre = 300000/1.5
        l = (Point_Terre.distance(self, other)*10**(-3))/cfibre
        return l

    def latencesat(self, other, sat):
        satcart = sat.cartesien()
        c = 3*10**8
        l = (vecteurs.norme(self-satcart) + vecteurs.norme(other-satcart))/c
        return l


# poleNspher=Point_Terre([6371000,81,-110])
# PNcart=poleNspher.cartesien()
# poleSspher=Point_Terre([6371000,-64.6,-138.30])
# PScart=poleSspher.cartesien()
rT = 6371*10**3
PALL = Point_Terre([rT, 48.866667, 2.333333])
NYLL = Point_Terre([rT, 40.779897, -73.968565])
BALL = Point_Terre([rT, -38.4212955, -63.587402499999996])
PSAT = Point_Terre([36000, 0, -30])

NYcart = NYLL.cartesien()
PAcart = PALL.cartesien()
NYspher = NYLL.spherique()
PAspher = PALL.spherique()
print(NYspher)
print(PAspher)
print(NYcart)
print(NYLL.cartesien())
print(PAcart)
# passez en degreq : *360/(2*np.pi)
d = Point_Terre.distance(NYLL.cartesien(), PALL.cartesien())
print("d= ", d/10**3, "km")
print()
# NYcart.displayp()
# NYcyl.displayp()
# passez en degreq : *360/(2*np.pi)


def affi():

    fenetre = tk.Tk()
    fenetre.geometry("1050x250")
    text = tk.Label(fenetre, text="Résultat ville 1")
    text.grid(column=0, row=0)
    text = tk.Label(fenetre, text="Résultat ville 2")
    text.grid(column=1, row=0)
    text = tk.Label(fenetre, text="Résultat ville 3")
    text.grid(column=2, row=0)
    text = tk.Label(fenetre, text="Coordonnées ville 1")
    text.grid(column=3, row=0)
    text = tk.Label(fenetre, text="Coordonnées ville 2")
    text.grid(column=4, row=0)
    text = tk.Label(fenetre, text="Coordonnées ville 3")
    text.grid(column=5, row=0)
    text = tk.Label(fenetre, text="Distance V1-V2")
    text.grid(column=6, row=0)
    text1 = tk.Label(fenetre, text="? km")
    text1.grid(column=6, row=1)
    text = tk.Label(fenetre, text="Distance V1-V3")
    text.grid(column=6, row=2)
    text2 = tk.Label(fenetre, text="? km")
    text2.grid(column=6, row=3)

    srtx = tk.Label(fenetre, text="x ville 1")
    srtx.grid(column=0, row=2)
    srty = tk.Label(fenetre, text="y ville 1")
    srty.grid(column=0, row=3)
    srtz = tk.Label(fenetre, text="z ville 1")
    srtz.grid(column=0, row=4)

    srtx1 = tk.Label(fenetre, text="x ville 2")
    srtx1.grid(column=1, row=2)
    srty1 = tk.Label(fenetre, text="yville 2")
    srty1.grid(column=1, row=3)
    srtz1 = tk.Label(fenetre, text="z ville 2")
    srtz1.grid(column=1, row=4)

    srtx2 = tk.Label(fenetre, text="x ville 3")
    srtx2.grid(column=2, row=2)
    srty2 = tk.Label(fenetre, text="yville 3")
    srty2.grid(column=2, row=3)
    srtz2 = tk.Label(fenetre, text="z ville 3")
    srtz2.grid(column=2, row=4)

    im1 = tk.Label(fenetre, bg="black", text="N-Y", fg="white")
    im1.grid(column=3, row=1)
    im2 = tk.Label(fenetre, bg="red", text="Paris", fg="white")
    im2.grid(column=4, row=1)
    im3 = tk.Label(fenetre, bg="blue", text="Buenos Aires", fg="white")
    im3.grid(column=5, row=1)

    xentr = tk.Entry(fenetre, width=10)
    xentr.grid(column=3, row=2)
    yentr = tk.Entry(fenetre, width=10)
    yentr.grid(column=3, row=3)
    zentr = tk.Entry(fenetre, width=10)
    zentr.grid(column=3, row=4)

    xentr1 = tk.Entry(fenetre, width=10)
    xentr1.grid(column=4, row=2)
    yentr1 = tk.Entry(fenetre, width=10)
    yentr1.grid(column=4, row=3)
    zentr1 = tk.Entry(fenetre, width=10)
    zentr1.grid(column=4, row=4)

    xentr2 = tk.Entry(fenetre, width=10)
    xentr2.grid(column=5, row=2)
    yentr2 = tk.Entry(fenetre, width=10)
    yentr2.grid(column=5, row=3)
    zentr2 = tk.Entry(fenetre, width=10)
    zentr2.grid(column=5, row=4)

    OptionList = ["Cartésien", "Sphérique", "Cylindrique"]
    variable = tk.StringVar(fenetre)
    variable.set(OptionList[0])
    opt = tk.OptionMenu(fenetre, variable, *OptionList)
    opt.config(width=7)  # ,font=('Helvetica', 12))
    opt.grid(column=0, row=1)

    variable1 = tk.StringVar(fenetre)
    variable1.set(OptionList[0])
    opt = tk.OptionMenu(fenetre, variable1, *OptionList)
    opt.config(width=7)  # ,font=('Helvetica', 12))
    opt.grid(column=1, row=1)

    variable2 = tk.StringVar(fenetre)
    variable2.set(OptionList[0])
    opt = tk.OptionMenu(fenetre, variable2, *OptionList)
    opt.config(width=7)  # ,font=('Helvetica', 12))
    opt.grid(column=2, row=1)

    def clicked1():
        x = float(xentr.get())
        y = float(yentr.get())
        z = float(zentr.get())

        x1 = float(xentr1.get())
        y1 = float(yentr1.get())
        z1 = float(zentr1.get())

        x2 = float(xentr2.get())
        y2 = float(yentr2.get())
        z2 = float(zentr2.get())

        nouvp = Point_Terre([x, y, z])
        nouvp1 = Point_Terre([x1, y1, z1])
        nouvp2 = Point_Terre([x2, y2, z2])

        if variable.get() == OptionList[0]:
            nouvpc = nouvp.cartesien()
        if variable1.get() == OptionList[0]:
            nouvpc1 = nouvp1.cartesien()
        if variable2.get() == OptionList[0]:
            nouvpc2 = nouvp2.cartesien()

        if variable.get() == OptionList[1]:
            nouvpc = nouvp.spherique()
        if variable1.get() == OptionList[1]:
            nouvpc1 = nouvp1.spherique()
        if variable2.get() == OptionList[1]:
            nouvpc2 = nouvp2.spherique()

        if variable.get() == OptionList[2]:
            nouvpc = nouvp.cylindrique()
        if variable1.get() == OptionList[2]:
            nouvpc1 = nouvp1.cylindrique()
        if variable2.get() == OptionList[2]:
            nouvpc2 = nouvp2.cylindrique()

        srtx.configure(text="{}".format(nouvpc[0]))
        srty.configure(text="{}".format(nouvpc[1]))
        srtz.configure(text="{}".format(nouvpc[2]))

        srtx1.configure(text="{}".format(nouvpc1[0]))
        srty1.configure(text="{}".format(nouvpc1[1]))
        srtz1.configure(text="{}".format(nouvpc1[2]))

        srtx2.configure(text="{}".format(nouvpc2[0]))
        srty2.configure(text="{}".format(nouvpc2[1]))
        srtz2.configure(text="{}".format(nouvpc2[2]))

        text1.configure(text="{} km".format(
            round(Point_Terre.distance(nouvp.cartesien(), nouvp1.cartesien())/10**3, 2)))
        text2.configure(text="{} km".format(
            round(Point_Terre.distance(nouvp1.cartesien(), nouvp2.cartesien())/10**3, 2)))
        f = open("resultats.txt", "w")

        nouvpca = nouvp.cartesien()
        nouvpsp = nouvp.spherique()
        nouvpcy = nouvp.cylindrique()

        nouvpca1 = nouvp1.cartesien()
        nouvpsp1 = nouvp1.spherique()
        nouvpcy1 = nouvp1.cylindrique()

        nouvpca2 = nouvp2.cartesien()
        nouvpsp2 = nouvp2.spherique()
        nouvpcy2 = nouvp2.cylindrique()

        f.write("coordonnées VILLE 1 \n\n")
        f.write("coordonnées Cartesiens : {} \n".format(nouvpca.displayr()))
        f.write("coordonnées Cylindriques : {} \n".format(nouvpcy.displayr()))
        f.write("coordonnées Spheriques : {} \n\n\n".format(nouvpsp.displayr()))

        f.write("coordonnées VILLE 2 \n\n")
        f.write("coordonnées Cartesiens : {} \n".format(nouvpca1.displayr()))
        f.write("coordonnées Cylindriques : {} \n".format(nouvpcy1.displayr()))
        f.write("coordonnées Spheriques : {} \n\n\n".format(nouvpsp1.displayr()))

        f.write("coordonnées VILLE 3 \n\n")
        f.write("coordonnées Cartesiens : {} \n".format(nouvpca2.displayr()))
        f.write("coordonnées Cylindriques : {} \n".format(nouvpcy2.displayr()))
        f.write("coordonnées Spheriques : {} \n\n\n".format(nouvpsp2.displayr()))

        f.write("latence V1-V2 (Paris-N.Y.C) sat = {} s \n".format(
            Point_Terre.latencesat(nouvpca1, nouvpca, PSAT)))
        f.write("latence V1-V2 (Paris-N.Y.C) fibr = {} s \n".format(
            Point_Terre.latencefibre(nouvpca1, nouvpca)))
        f.write("dist V1-V2 (Paris-N.Y.C) = {} km \n\n".format(
            round(Point_Terre.distance(nouvpca, nouvpca1)/10**3, 2)))

        f.write("latence V1-V3 (Paris-Buenos aire) sat = {} s \n".format(
            Point_Terre.latencesat(nouvpca2, nouvpca, PSAT)))
        f.write("latence V1-V3 (Paris-Buenos aire) fibr = {} s \n".format(
            Point_Terre.latencefibre(nouvpca2, nouvpca)))
        f.write("dist V1-V3 (Paris-Buenos aire) = {} km".format(
            round(Point_Terre.distance(nouvpca, nouvpca2)/10**3, 2)))
        f.close()

    def clicked2():
        x = float(xentr.get())
        y = float(yentr.get())
        z = float(zentr.get())

        x1 = float(xentr1.get())
        y1 = float(yentr1.get())
        z1 = float(zentr1.get())

        x2 = float(xentr2.get())
        y2 = float(yentr2.get())
        z2 = float(zentr2.get())

        nouvp2 = Point_Terre([x2, y2, z2])
        nouvp1 = Point_Terre([x1, y1, z1])
        nouvp = Point_Terre([x, y, z])
        point = nouvp.cartesien()
        point1 = nouvp1.cartesien()
        point2 = nouvp2.cartesien()
        Point_Terre.generimage(point, point1, point2)

    def clicked3(a=NYLL, b=PALL, c=BALL):
        xentr.delete(0, "end")
        xentr.insert(0, a[0])
        yentr.delete(0, "end")
        yentr.insert(0, a[1])
        zentr.delete(0, "end")
        zentr.insert(0, a[2])

        xentr1.delete(0, "end")
        xentr1.insert(0, b[0])
        yentr1.delete(0, "end")
        yentr1.insert(0, b[1])
        zentr1.delete(0, "end")
        zentr1.insert(0, b[2])

        xentr2.delete(0, "end")
        xentr2.insert(0, c[0])
        yentr2.delete(0, "end")
        yentr2.insert(0, c[1])
        zentr2.delete(0, "end")
        zentr2.insert(0, c[2])

    btn1 = tk.Button(fenetre, text="Calculer", bg="grey",
                     fg="white", command=clicked1)
    btn1.grid(column=7, row=0)
    btn2 = tk.Button(fenetre, text="Afficher", bg="grey",
                     fg="white", command=clicked2)
    btn2.grid(column=7, row=1)
    btn3 = tk.Button(fenetre, text="preset", bg="grey",
                     fg="white", command=clicked3)
    btn3.grid(column=7, row=2)

    fenetre.mainloop()


affi()
