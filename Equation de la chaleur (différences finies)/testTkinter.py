# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 18:32:12 2022

@author: vince
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
import tkinter as tk
from pylab import *

def Code(U0,f):   #Definition des parametres initiales
    N=100
    M=30
    T=0.25
    L=1
    dt=T/N
    dx=L/(M+1)
    dy=L/(M+1)
    gamma=0.05
    alpha=gamma*dt/(dx)**2
    x=np.linspace(dx,L-dx,M)
    y=np.linspace(dx,L-dx,M)
    k=0
    
    def Matrix_Laplace(N=10,A=4,B=-1,C=-1): 
        M=B*np.eye(N)+A*np.diag(np.ones(N-1),1)+C*np.diag(np.ones(N-1),-1)
        return M 

#On definit toutes les matrices de notre schema numérique

    A=Matrix_Laplace(M,-alpha,4*alpha+(3/2),-alpha)-alpha*np.eye(M,k=1)-alpha*np.eye(M,k=-1)
    P=-alpha*np.eye(M)
    B=np.kron(np.eye(M),A)+np.kron(np.eye(M,k=1),P)+np.kron(np.eye(M,k=-1),P)
    B_inv=np.linalg.inv(B)
    
#On definit nos conditions limites

    CL=np.zeros((M*M,1))
    C=Matrix_Laplace(M,1,-2,1)

#Definition des matrices U0 et U1 (par dev de taylor)

    U1=np.zeros((n+1,n+1))  
    Un1=np.reshape(Un0+dt*(f-gamma*(C@(Un0/(dx**2))+C@(Un0/(dy**2)))),(M*M,1))
    Un0=np.reshape(Un0,(M*M,1))
    f=np.reshape(f,(M*M,1))
    
    
    X,Y=np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    
    
    Uijn=np.zeros((M,M,N))
    
#Definition des matrices au rang n+2
    for n in range (N):
        Un2=dt*B_inv@ f+2*B_inv@Un1-0.5*B_inv@Un0
        Un2.reshape(M,M)[0,:],Un2.reshape(M,M)[:,0],Un2.reshape(M,M)[M-1,:],Un2.reshape(M,M)[:,M-1]=0,0,0,0
        Un0=Un1
        Un1=Un2
        Uijn[:,:,n]=np.reshape(Un2,(M,M))
        
     
    #Adding the colorbar 
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    m.set_array(np.max(Uijn))
    cbar = plt.colorbar(m)
        
    def animate(i):
        global k
        Z = Uijn[:,:,k]
        k += 1
        ax.clear()
        ax.plot_surface(X,Y,Z,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
        #ax.contour(X,Y,Z)
        ax.set_zlim(0,np.max(Uijn))
        
    anim = animation.FuncAnimation(fig,animate,frames=220,interval=20)
    plt.show()
    
    #Uij1=np.reshape(Uij0+dt*(f3(x,y)-gamma*(C@(Uij0/(dx**2))+C@(Uij0/(dy**2)))),(N*N,1))
    #Uij0=np.reshape(Uij0,(N*N,1))
    


#Fenetre
def lancelecailloubro():
    window = tk.Tk()
    window.title("Graphe des Ondulations")
    '''
    w = window.winfo_width() 
    h = window.winfo_height()
    '''
    window.maxsize(width=900, height=400)
    window.minsize(width=900, height=400)
    texte_help = tk.Label(text = "Bienvenue sur notre application !\n Vous pourrez ici observer la diffusion d'une onde dans l'eau en 3D. \n Nous vous souhaitons une bonne utilisation.")
    texte_help.grid(row = 5, column= 3)
    
    '''
    #Insertion des conditions initiales
    texte_saisie1 = tk.Label(window, text = "Conditions initiales:")
    texte_saisie1.grid(row = 18, column = 0)
    '''
    
    cond1 = tk.Label(window, text = "Choisissez vos conditions initiales :")
    cond1.grid(column=2, row=10)
    coin1 = tk.Label(window, text = "   ")
    coin1.grid(column=0, row=0)
    coin2 = tk.Label(window, text = "   ")
    coin2.grid(column=20, row=0)
    coin3 = tk.Label(window, text = "   ")
    coin3.grid(column=20, row=20)
    coin4 = tk.Label(window, text = "   ")
    coin4.grid(column=0, row=20)
    '''
    entree1 = tk.Entry(window, textvariable = C1)
    entree1.grid(row = 21, column = 0) 
    
    cond2 = tk.Label(window, text = "Condition initiale 2:")
    cond2.grid(row = 22, column = 0)
    entree2 = tk.Entry(window, textvariable = C2)
    entree2.grid(row = 23, column = 0)
    
    cond3 = tk.Label(window, text = "Condition initiale 3:")
    cond3.grid(row = 24, column = 0)
    entree3 = tk.Entry(window, textvariable = C3)
    entree3.grid(row = 25, column = 0)
    
    cond4 = tk.Label(window, text = "Condition initiale 3:")
    cond4.grid(row = 26, column = 0)
    entree4 = tk.Entry(window, textvariable = C4)
    entree4.grid(row = 27, column = 0)
    '''
    #    Menu déroulant
    OptionList=["Point d'appui","Barre chauffante"]  
    variable=tk.StringVar(window)
    variable.set(OptionList[0])
    opt=tk.OptionMenu(window, variable, *OptionList)
    opt.config(width=20) #,font=('Helvetica', 12))
    opt.grid(column=2, row=11)
#problème : La barre et le point sur superposent quand on refait plusieurs fois
    def clicked1():
        plt.close()
        N=100
        M=30
        T=0.25
        L=1
        dt=T/N
        dx=L/(M+1)
        dy=L/(M+1)
        gamma=0.05
        alpha=gamma*dt/(dx)**2
        x=np.linspace(dx,L-dx,M)
        y=np.linspace(dx,L-dx,M)
        k=0
        def f_exi2(x,y):
            global M
            F=np.zeros((M,M))
            for i in range(M):
                for j in range(M):
                    F[i,j]=np.sin(x[i])+np.cos(y[j])
            return F
          
        def f_exi3(x,y):
            global M
            F=np.zeros((M,M))
            for i in range(-4+M//2,4+M//2):
                for j in range(-4+M//2,4+M//2):
                    F[i,j]=2
            return F
        
        def f_exi(x,y):
            return np.zeros((M,M))
        
        if variable.get() == OptionList[0]:
           Un0=np.ones((M,M))
           f=np.zeros((M,M))
           for i in range(0,M) :
               for j in range(0,M) :
                   Un0[i,j]=4*np.abs(np.sin(2*dx*(i +2)*np.pi/L)+np.sin(2*dx*(j +2)*np.pi/L)+np.sin(2*dx*(i*j +2)*np.pi/L))
           Code(Un0, f)
             
        if variable.get() == OptionList[1]:
           Un0=np.ones((M,M))  
           f=np.zeros((M,M))
           for i in range(0,M):
               for j in range(0,M) :
                   Un0[i,j]=np.abs(i*dx*(L-i*dx)*j*dx*(L-j*dx))/(4*L)
           Code(Un0,f)
                
        
    # Affichage graphique
    bouton=tk.Button(window, text = "Afficher le graphique",command=clicked1) #command=fenetre.quit)
    bouton.grid(row = 19, column = 3, padx = 50)
    
    
    window.mainloop()
lancelecailloubro()