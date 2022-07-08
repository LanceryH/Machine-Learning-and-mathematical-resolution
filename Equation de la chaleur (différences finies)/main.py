# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation


"""
________________________________________________________________________________________________________________________
____________________________________________/        \__________________________________________________________________
___________________________________________| Partie 2 |_________________________________________________________________
____________________________________________\________/__________________________________________________________________

"""

#Definition des parametres initiales

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

#Permet de creer des matrices avec coeffs sur diag sup et diag inf
def Matrix_Laplace(N=10,A=4,B=-1,C=-1):
    M=B*np.eye(N)+A*np.diag(np.ones(N-1),1)+C*np.diag(np.ones(N-1),-1)
    return M 


#implementation des differentes fonctions  à choisir

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

#Creation des differentes matrices

A=Matrix_Laplace(M,-alpha,4*alpha+(3/2),-alpha)-alpha*np.eye(M,k=1)-alpha*np.eye(M,k=-1)
P=-alpha*np.eye(M)
B=np.kron(np.eye(M),A)+np.kron(np.eye(M,k=1),P)+np.kron(np.eye(M,k=-1),P)
B_inv=np.linalg.inv(B)

#On defini les conditions aux limites
CL=np.zeros((M*M,1))

C=Matrix_Laplace(M,1,-2,1) #Matrice de Laplace necessaire pour U1

f=f_exi(x,y)

Un0=np.ones((M,M))
Un1=np.zeros((M,M))

#for i in range(0,M) :
#    for j in range(0,M) :
#        Un0[i,j]=4*np.abs(np.sin(2*dx*(i +2)*np.pi/L)+np.sin(2*dx*(j +2)*np.pi/L)+np.sin(2*dx*(i*j +2)*np.pi/L))
#for i in range(0,M):
#    for j in range(0,M) :
#        Un0[i,j]=np.abs(i*dx*(L-i*dx)*j*dx*(L-j*dx))/(4*L)*

#Definition des matrices U0 et U1 (par dev de taylor)

Un1=np.reshape(Un0+dt*(f-gamma*(C@(Un0/(dx**2))+C@(Un0/(dy**2)))),(M*M,1))
Un0=np.reshape(Un0,(M*M,1))
f=np.reshape(f,(M*M,1))


X,Y=np.meshgrid(x,y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')



Uijn=np.zeros((M,M,N))

#On determine la matrice u a chaque rang de n
 
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
    
#On plot les resultats

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

