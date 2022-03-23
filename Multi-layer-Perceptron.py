# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:15:33 2021

@author: Emilio
"""
import numpy as np
np.random.seed(2)
x = np.array([
[0, 0, 0, 0], #Matriz de entradas
[0, 0, 0, 1],
[0, 1, 1, 0],
[0, 1, 1, 1],
[1, 0, 0, 0],
[1, 1, 1, 0]], dtype = float)

d= np.array([
[0,0],
[1,1],
[1,1],
[0,1],
[1,0],
[1,0]], dtype= float)

N= 4
L= 6
M= 2
Q= 6
wh= np.random.random([L,N])*2-1
wo= np.random.random([M,L])*2-1
E=1

while E>= 0.0001:
    for i in range(Q):
    # Forward
        
        neth= np.matmul(wh,x[i].T)
        yh= 1/(1+np.exp(-neth))
        neto= np.matmul(wo,yh)
        y= 1/(1+np.exp(-neto))
       
    
    # Backward
    
        so= ((d[i].T)-y)*(y*(1-y))
        sh=  yh*(1-yh)*[np.matmul(wo.T,so)]
        alpha= 0.5
        aso= alpha*so[np.newaxis].T
        yht= yh[np.newaxis]
        dwO= np.matmul(aso,yht)
        wo+= dwO
        
        ash= (alpha*sh).T
        xx= x[i][np.newaxis]
        dwh= np.matmul(ash,xx)
        wh+= dwh

# error
        E= max(abs(so))
       


# TEST

xt= np.array([
[0,0,0,1],
[0,1,1,0],
[1,0,1,0],
[1,0,0,1],
[1,1,1,1]],dtype= float)

for i in range(5):
    # Forward
    xt_predict= xt[i,:]
    neth_t= np.matmul(wh,xt_predict.T)
    yh_t= 1/(1+np.exp(-2*neth_t))
    neto_t= np.matmul(wo,yh_t)
    y_t= 1/(1+np.exp(-2*neto_t))
    print('prediccion',xt_predict,y_t)
