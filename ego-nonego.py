
# coding: utf-8

# In[77]:


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


# In[78]:


def comp(a,b):
	if(int(a[6:-4])>=int(b[6:-4])):
		return 1
	else:
		return -1


# In[79]:


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
network.summary()
network.load_weights("optical_flow_deep.h5")

# In[81]:


rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
network.compile(loss='mean_squared_error', optimizer=rms)


# In[100]:

for rv in range(11,12):
	original_matrix=[]
	transformed_matrix=[]


	# In[101]:


	for k in range(1,3):
	    path_major='Data/'+str(k)
	    for j in range(rv*250+1,(rv+1)*250):
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
	    for j in range(rv*250+1,(rv+1)*250):
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
	print('Images of '+str(rv)+'th are loaded')


	# In[102]:


	data,Label = shuffle(original_matrix,transformed_matrix, random_state=2)
	X_train,X_test,Y_train,Y_test=train_test_split(data, Label, test_size=0.25, random_state=2)
	del data[:]
	del Label[:]
	del original_matrix[:]
	del transformed_matrix[:]
	X_train=array(X_train)
	Y_train=array(Y_train)
	X_test=array(X_test)
	Y_test=array(Y_test)


	# In[103]:


	X_train = X_train.astype('float32') / 255.
	X_test = X_test.astype('float32') / 255.
	Y_train = Y_train.astype('float32') / 255.
	Y_test = Y_test.astype('float32') / 255.
	print('Trainset-'+ str(X_train.shape[0]))
	print('Testset-' + str(X_test.shape[0]))

	batch_size=4
	num_batches=math.ceil(len(X_train)/batch_size)	
	# In[112]:


	for epoch in range(70):
							batch_size=4
							num_batches=93	
							X_train,Y_train=shuffle(X_train,Y_train)
							print ("Epoch is: %d\n" % epoch)
							print ("Number of batches: %d\n" % int(num_batches))
							for batch in range(int(num_batches)):
								batch_train_X=X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
								batch_train_Y=Y_train[batch*batch_size:min((batch+1)*batch_size,len(Y_train))]
								loss=network.train_on_batch(batch_train_X,batch_train_Y)
								print ('epoch_num: %d batch_num: %d train loss: %f\n' % (epoch,batch,loss))
							network.save_weights("optical_flow_deep.h5")
							if(epoch%3==0):
								X_test,Y_test=shuffle(X_test,Y_test)
								decoded_imgs=network.predict(X_test[:1])                            
								temp = np.zeros([240, 320*3,3])
								temp[:, :320,:1] = X_test[0,:,:,:1]
								temp[:, 320:320*2,:1] = Y_test[0,:,:,:1]
								temp[:, 320*2:,:1] = decoded_imgs[0,:,:,:1]
								# temp[:,:,1]=temp[:,:,0]
								# temp[:,:,2]=temp[:,:,0]
								temp[:, :320,1:2] = X_test[0,:,:,1:2]
								temp[:, 320:320*2,1:2] = Y_test[0,:,:,1:2]
								temp[:, 320*2:,1:2] = decoded_imgs[0,:,:,1:2]

								temp[:, :320,2:3] = X_test[0,:,:,2:3]
								temp[:, 320:320*2,2:3] = Y_test[0,:,:,2:3]
								temp[:, 320*2:,2:3] = decoded_imgs[0,:,:,2:3]


								temp = temp*255
								scipy.misc.imsave('res/' +str(rv) + '_'+ str(epoch) + ".jpg", temp)
								num_batches=math.ceil(len(X_test)/batch_size)
								for batch in range(int(num_batches)):
									batch_test_X=X_test[batch*batch_size:min((batch+1)*batch_size,len(X_test))]
									batch_test_Y=Y_test[batch*batch_size:min((batch+1)*batch_size,len(Y_test))]
									loss+=network.test_on_batch(batch_test_X,batch_test_Y)
								loss/=num_batches	
								print ('epoch_num: %d batch_num: %d test loss: %f\n' % (epoch,batch,loss))
								
	
	loss=0
	X_test,Y_test=shuffle(X_test,Y_test)
	num_batches=math.ceil(len(X_test)/batch_size)
	for batch in range(int(num_batches)):
							batch_test_X=X_test[batch*batch_size:min((batch+1)*batch_size,len(X_test))]
							batch_test_Y=Y_test[batch*batch_size:min((batch+1)*batch_size,len(Y_test))]
							loss+=network.test_on_batch(batch_test_X,batch_test_Y)
	loss/=num_batches	
	print ('epoch_num: %d batch_num: %d test loss: %f\n' % (epoch,batch,loss))
	X_train=None
	Y_train=None
	X_test=None
	Y_test=None
	
	# In[111]:



