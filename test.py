from keras.layers import Input, Dense , merge
from keras.layers.core import Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop
import os
import os.path
import numpy as np
import numpy 
from PIL import Image
from numpy import * 
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import scipy.misc
import math
from scipy import signal

input_img = Input(shape=(240,320,9))  # adapt this if using `channels_first` image data format
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(input_img)  # 160x180x16
x=BatchNormalization()(x)
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(input_img)  # 160x180x16
x=BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x) # 80x90x16
x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)  
x=BatchNormalization()(x)
x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)  
x=BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)   # 40x45x8
x = Convolution2D(256, (3, 3), activation='relu', padding='same')(x) 
x=BatchNormalization()(x)
x = Convolution2D(256, (3, 3), activation='relu', padding='same')(x) 
x=BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)  #20x23x8
x = Convolution2D(512, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same',name='encoded')(x) 


# In[80]:


x=UpSampling2D((2,2))(encoded)
x = Convolution2D(512, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(256, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(256, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x=UpSampling2D((2,2))(x)
x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
decoded = Convolution2D(9, (3, 3), activation='sigmoid', padding='same')(x)
network=Model(input_img,decoded)
network.load_weights("optical_flow_deep.h5")
rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
network.compile(loss='mean_squared_error', optimizer=rms)

original_matrix=[]
transformed_matrix=[]


# In[101]:


for k in range(1,2):
    path_major='Data/'+str(k)
    for j in range(1,100):
	im=array(Image.open(path_major+"/"+str(j)+".jpg"))
	img=im
	new=np.zeros([img.shape[0],img.shape[1],9])
	for i in range(3):
	    new[:,:,i]=img[:,:,i]
	im=im.squeeze()#Remove acces dimension
	im=np.swapaxes(im,0,2)#Swap axes to feet thos of standart image (x,y,d)
	im=np.swapaxes(im,1,2)
	Gx=[[1,2,1], [0 , 0 ,0],[-1,-2,-1]]#Build sobel x,y gradient filters
	Gy=np.swapaxes(Gx,0,1)#Build sobel x,y gradient filters
	ndim=im[:,1,1].shape[0]# Get the depth (number of filter of the layer)
	TotGrad=np.zeros(im[1,:,:].shape) #The averge gradient map of the image to be filled later
	i=3
	for ii in range(ndim):# Go over all dimensions (filters) 
	   gradx = signal.convolve2d(im[ii,:,:],Gx,  boundary='symm',mode='same');#Get x sobel response of ii layer
	   grady = signal.convolve2d(im[ii,:,:],Gy,  boundary='symm',mode='same');#Get y sobel response of ii layer
	   new[:,:,ii+i]+=gradx
	   new[:,:,ii+i+1]+=grady
	   i+=1
	original_matrix.append(new)
    for j in range(1,100):
	im=array(Image.open(path_major+"/"+str(j)+"_OF.jpg"))
	img=im
	new=np.zeros([img.shape[0],img.shape[1],9])
	for i in range(3):
	    new[:,:,i]=img[:,:,i]
	im=im.squeeze()#Remove acces dimension
	im=np.swapaxes(im,0,2)#Swap axes to feet thos of standart image (x,y,d)
	im=np.swapaxes(im,1,2)
	Gx=[[1,2,1], [0 , 0 ,0],[-1,-2,-1]]#Build sobel x,y gradient filters
	Gy=np.swapaxes(Gx,0,1)#Build sobel x,y gradient filters
	ndim=im[:,1,1].shape[0]# Get the depth (number of filter of the layer)
	TotGrad=np.zeros(im[1,:,:].shape) #The averge gradient map of the image to be filled later
	i=3
	for ii in range(ndim):# Go over all dimensions (filters) 
	   gradx = signal.convolve2d(im[ii,:,:],Gx,  boundary='symm',mode='same');#Get x sobel response of ii layer
	   grady = signal.convolve2d(im[ii,:,:],Gy,  boundary='symm',mode='same');#Get y sobel response of ii layer
	   new[:,:,ii+i]+=gradx
	   new[:,:,ii+i+1]+=grady
	   i+=1
	transformed_matrix.append(new)
print('Images of are loaded')


# In[102]:


data,Label = shuffle(original_matrix,transformed_matrix, random_state=2)
del original_matrix[:]
del transformed_matrix[:]
X_test=array(data)
Y_test=array(Label)


# In[103]:


X_test = X_test.astype('float32') / 255.
Y_test = Y_test.astype('float32') / 255.
print('Testset-' + str(X_test.shape[0]))
tot_loss=0
for i in range(X_test.shape[0]):
		decoded_imgs=network.predict(X_test[i:i+1])         
		loss=network.test_on_batch(X_test[i:i+1],Y_test[i:i+1])
		tot_loss+=loss                   
		temp = np.zeros([240, 320*3,3])
		temp[:, :320,:1] = X_test[i,:,:,:1]
		temp[:, 320:320*2,:1] = Y_test[i,:,:,:1]
		temp[:, 320*2:,:1] = decoded_imgs[0,:,:,:1]
		# temp[:,:,1]=temp[:,:,0]
		# temp[:,:,2]=temp[:,:,0]
		temp[:, :320,1:2] = X_test[i,:,:,1:2]
		temp[:, 320:320*2,1:2] = Y_test[i,:,:,1:2]
		temp[:, 320*2:,1:2] = decoded_imgs[0,:,:,1:2]
		temp[:, :320,2:3] = X_test[i,:,:,2:3]
		temp[:, 320:320*2,2:3] = Y_test[i,:,:,2:3]
		temp[:, 320*2:,2:3] = decoded_imgs[0,:,:,2:3]
		temp = temp*255
		scipy.misc.imsave('res/' +str(i) + '_'+ str(loss*100) + ".jpg", temp)
print(tot_loss/X_test.shape[0])
# In[101]:



