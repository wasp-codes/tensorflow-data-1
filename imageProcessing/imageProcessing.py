#image segmentation master
#loading libraries 
from skimage.color import rgb2gray
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#to display the outputs inline on ipython
#%matplotlib inline

from scipy import ndimage

#read and plot the image
image = plt.imread('table.jpg')
plt.imshow(image)

#check the shape of the image
image.shape

#region based segmentation
#conversion of the three channel RGB image (544, 494,3) to a single channel grayscale
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')

#check the shape of the image
gray.shape

#The image shape (height, width, channels) can be used to find the threshold between 
#images and the background by using the mean of the height and width values
#pixel values greater than the threshold values indicate an object over background

gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 3
    elif gray_r[i] > 0.5:
        gray_r[i] = 2
    elif gray_r[i] > 0.25:
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')

#edge detection segmentation
#k-means clustering segmentation
 # dividing by 255 to bring the pixel values between 0 and 1
img = cv.imread('table.jpg')/255
pic = cv.resize(img,(255,255))
print(pic.shape)
plt.imshow(pic)

#k-means clustering demands that the image is a (height*width, channels) array
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape

#get clusters by fitting the k-means algorithm on this array
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

#convert the array into an image
cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)