from keras.layers import Input, Dense
from keras.layers.core import Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop
import os
import os.path
import numpy 
from PIL import Image
from numpy import * 
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def comp(a,b):
	if(int(a[6:-4])>=int(b[6:-4])):
		return 1
	else:
		return -1

######### convolutional encoder model

input_img = Input(shape=(240,320,3))  # adapt this if using `channels_first` image data format
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)  # 160x180x16
x=BatchNormalization()(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x) 
x=BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x) # 80x90x16
x=BatchNormalization()(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)  
x=BatchNormalization()(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)   # 40x45x8
x=BatchNormalization()(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x) 
x=BatchNormalization()(x)
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)  
x=BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)  #20x23x8
x=BatchNormalization()(x)
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same',name='encoded')(x) #10x12x8

# at this point the representation is (10,15,8) i.e. 1200-dimensional
x=UpSampling2D((2,2))(encoded)
x=BatchNormalization()(x)
x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x=BatchNormalization()(x)
x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x=BatchNormalization()(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x=BatchNormalization()(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
decoded = Convolution2D(3, (3, 3), activation='sigmoid', padding='same')(x)

ada=Adadelta(lr=5.0, rho=0.95, epsilon=1e-08, decay=0.001)
rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)

autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(loss='mean_squared_error', optimizer=rms)
autoencoder.summary()
exit()
original_matrix=[]
transformed_matrix=[]

out="opticalf_frames/"
inp="dataset/train_set/"

folder1 = os.listdir(out) #ego/non_ego

part_size = 512

n_part = 0

for i in range(len(folder1)):

	folder2 = os.listdir(out + folder1[i]) #datasets

	for j in range(len(folder2)):

		print("Loading " + folder2[j] + " dataset...")

		folder3 = os.listdir(out + folder1[i] + "/" + folder2[j]) #clips

		for k in range(len(folder3)):

			files = os.listdir(out + folder1[i] + "/" + folder2[j] + "/" + folder3[k])
			files = sorted(files)

			for l in range(1, len(files)):

				if len(original_matrix) == part_size:

					if n_part != 0 :
						print("Loading previous weights...")
						autoencoder.load_weights("weights/model_encoder_withbatchnorm_deep_tt12.h5")

					data,Label = shuffle(original_matrix,transformed_matrix, random_state=2)


					X_train, X_test, y_train, y_test = train_test_split(data, Label, test_size=0.25, random_state=2)

					X_test=array(X_test)
					X_test=X_test.reshape(X_test.shape[0],240,320,3)
					y_test=array(y_test)
					y_test=y_test.reshape(y_test.shape[0],240,320,3)

					#x_train = X_train.astype('float32') / 255.
					x_test = X_test.astype('float32') / 255.

					#y_train = y_train.astype('float32') / 255.
					y_test = y_test.astype('float32') / 255.

					print(x_test.shape)
					print(y_test.shape)

					########### train the encoder decoder network
					for epoch in range(2):
						train_X,train_Y=shuffle(X_train,y_train)
						print ("Epoch is: %d\n" % epoch)
						batch_size=64
						print ("Number of batches: %d\n" % int(len(train_X)/batch_size))
						num_batches=int(len(train_X)/batch_size)
						for batch in range(num_batches):
							batch_train_X=train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
							batch_train_Y=train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]

							batch_train_X=array(batch_train_X)
							batch_train_X=batch_train_X.reshape(batch_train_X.shape[0],240,320,3)
							batch_train_Y=array(batch_train_Y)
							batch_train_Y=batch_train_Y.reshape(batch_train_Y.shape[0],240,320,3)

							batch_train_X=batch_train_X.astype('float32') / 255.
							batch_train_Y=batch_train_Y.astype('float32') / 255.

							loss=autoencoder.train_on_batch(batch_train_X,batch_train_Y)
							print ('epoch_num: %d batch_num: %d loss: %f\n' % (epoch,batch,loss))
						autoencoder.save_weights("weights/model_encoder_withbatchnorm_deep_tt12.h5")
						x_test,y_test=shuffle(x_test,y_test)
						decoded_imgs=autoencoder.predict(x_test[:15])

						# if(epoch%10==0):
		
						# 	n = 10 # how many digits we will display
						# 	plt.figure(figsize=(3200, 720))
						# 	for i in range(n):
						# 	# display original
						# 		ax = plt.subplot(3, n, i + 1)
						# 		ax.get_xaxis().set_visible(False)
						# 		ax.get_yaxis().set_visible(False)
						# 		plt.imshow(x_test[i])
							 
						# 		ax = plt.subplot(3, n, i + 1+n)
						# 		ax.get_xaxis().set_visible(False)
						# 		ax.get_yaxis().set_visible(False)
						# 		plt.imshow(y_test[i])
							
						# 		ax = plt.subplot(3, n, i + 1 + n + n)
						# 		ax.get_xaxis().set_visible(False)
						# 		ax.get_yaxis().set_visible(False)
						# 		plt.imshow(decoded_imgs[i])
							
						# 	plt.savefig("results/" + str(n_part) + "%#06d.jpg" % (epoch))
						# 	plt.close()

						del original_matrix[:]
						del transformed_matrix[:]

						n_part = n_part + 1

				opticalf = array(Image.open(out + folder1[i] + "/" + folder2[j] + "/" + folder3[k] + "/" + files[l]))
				transformed_matrix.append(opticalf)

				stereo = array(Image.open(inp + folder1[i] + "/" + folder2[j] + "/" + folder3[k] + "/" + files[l-1]))
				original_matrix.append(stereo)

		print(folder2[j] + " loaded")
