# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:51:29 2020

@author: 20172671mengqi
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
 
 
'''连续读入20张图片，生成样本矩阵 '''
s='TestDatabase/TrainDatabase/'
img=cv2.imread('TestDatabase/TrainDatabase/1.jpg',cv2.IMREAD_GRAYSCALE)#读入第一张图片
X=img.reshape(80*120,1) #维度变换
for i in range(2,21):
    path=s+'%d' %i+'.jpg'
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img=img.reshape(80*120,1)
    X=np.hstack((X,img)) #样本组合
M=np.mean(X,axis=1)
X=(X.T-M.T).T

'''求基向量P'''
def P_X(X,k):
    C=np.dot(X,(X.T)) #求协方差矩阵
    a,b=np.linalg.eig(C) #求特征值和特征向量
    k=5
    P=b[np.argsort(a,axis=0)[:-k-1:-1],:] #求前k大的特征值对对应的特征向量
    return P


'''求两张脸的欧式距离'''
def face_dist(img1,img2,P):
    img1=img1.reshape(80*120,1)
    img2=img2.reshape(80*120,1)
    y1=np.dot(P,img1)
    y2=np.dot(P,img2)
    dist=np.linalg.norm(y1-y2)
    return dist


'''寻找欧式距离最小的脸'''
test_img=cv2.imread('6.jpg',cv2.IMREAD_GRAYSCALE)
min_dist=100000000000
min_dist_i=0
P=P_X(X,6)
for i in range(1,21):  #求最小欧式距离对应的图片
    path=s+'%d' %i+'.jpg'
    train_img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)    
    dist=face_dist(test_img,train_img,P)
    if(dist<min_dist):
        min_dist=dist
        min_dist_i=i
result_img_path=s+'%d' %min_dist_i+'.jpg' #结果图片路径
result_img=cv2.imread(result_img_path,cv2.IMREAD_GRAYSCALE)
imgs=np.hstack([test_img,result_img])
cv2.imshow("mutil_pic",imgs)
cv2.waitKey(0)



 