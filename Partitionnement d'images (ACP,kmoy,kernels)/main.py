# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:35:17 2022

@author: hugol
"""

from mabiblio import *

image=cv2.imread("feuille.jpg")
nbpts=25 #nombre de points random par ligne
n,m,p=np.shape(image)
k=2
data_pix=data_pixels(image, choixpts(image, nbpts)[1],nbpts)

ll,llind=Kmoy(ACPimg(data_pix,q=2),k ,epsilon=10**(-10))    


imgmasqueponctuel=Masque(llind,image,nbpts,data_pix)    
    
cv2.imwrite("resultat1.jpg", imgmasqueponctuel)     


imgmasque=RemplissageMasque(imgmasqueponctuel.astype('uint8') * 255)

cv2.imwrite("resultat2.jpg",imgmasque) 

imgmasquetot1,imgmasquetot2=Masquetot(image,imgmasque)  

cv2.imwrite("resultat3.jpg", imgmasquetot1)

cv2.imwrite("resultat4.jpg", imgmasquetot2)
#%%

image=cv2.imread("feuille.jpg")
nbpts=25 #nombre de points random par ligne
n,m,p=np.shape(image)

k=2
data_pix=data_pixels(image, choixpts(image, nbpts)[1],nbpts)
w=np.eye(np.shape(data_pix)[1])
for i in range (8):
    w[i,i]=w[i,i]*i*10

#w[5,5]=0
ll,llind=Kmoy(ACPond(data_pix,2,w),k ,epsilon=10**(-10))    


imgmasqueponctuel=Masque(llind,image,nbpts,data_pix)    
    
cv2.imwrite("resultatpond1.jpg", imgmasqueponctuel)     


imgmasque=RemplissageMasque(imgmasqueponctuel.astype('uint8') * 255)

cv2.imwrite("resultatpond2.jpg",imgmasque) 

imgmasquetot1,imgmasquetot2=Masquetot(image,imgmasque)  

cv2.imwrite("resultatpond3.jpg", imgmasquetot1)

cv2.imwrite("resultatpond4.jpg", imgmasquetot2)