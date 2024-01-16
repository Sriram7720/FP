import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

from google.colab import drive
drive.mount('/content/drive')

def normalize(v):
    return v / np.sqrt(v.dot(v))

def generate_phi():
    np.random.seed(333)
    phi = np.random.normal(size=(272, 1089))
    n = len(phi)

    # perform Gramm-Schmidt orthonormalization

    phi[0, :] = normalize(phi[0, :])

    for i in range(1, n):
        Ai = phi[i, :]
        for j in range(0, i):
            Aj = phi[j, :]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        phi[i, :] = normalize(Ai)

    return phi

def divide_img_to_blocks(image, stride=14, filter_size=33):
    (h,w) = image.shape
    image = image[0:h-h%3, 0:w-w%3]
    (h,w) = image.shape
    h_iters = ((h - filter_size) // stride) + 1
    w_iters = ((w - filter_size) // stride) + 1
    blocks = []
    for i in range(h_iters):
        for j in range(w_iters):
            blocks.append(image[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j])

    return np.asarray(blocks)

#%% CREATE DATASET

# Read data and create patches
image_dataset = []
pics = sorted(os.listdir("/content/drive/MyDrive/LR"))
for pic in pics:
    image= cv2.imread("/content/drive/MyDrive/LR"+'/'+pic)
    image_lum = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image_lum = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_lum = image_lum[:,:,0]
    blocks = divide_img_to_blocks(image_lum)
    image_dataset.append(blocks)

image_dataset = np.concatenate(image_dataset, axis=0)

#A = scipy.io.loadmat("/content/drive/MyDrive/phi_0_25_1089.mat")["phi"]

A = generate_phi()

labels = np.copy(image_dataset)

X_train, X_test, y_train, y_test = train_test_split(image_dataset, image_dataset, test_size=0.1, random_state=333)

X_test.shape

train_dataset=tf.Variable(X_train,dtype=tf.float32)
test_dataset=tf.Variable(X_test,dtype=tf.float32)
train_label=tf.Variable(y_train,dtype=tf.float32)
test_label=tf.Variable(y_test,dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
dataset_y=tf.data.Dataset.from_tensor_slices(train_label)
test_x= tf.data.Dataset.from_tensor_slices(test_dataset)
test_y=tf.data.Dataset.from_tensor_slices(test_label)

#dataset = dataset.batch(batch_size=128,drop_remainder=True)

len(dataset)

#tf.compat.v1.disable_eager_execution()


A1 = tf.Variable(A.T, dtype=tf.float32)

#x = tf.compat.v1.placeholder(tf.float32, shape=[None, 33, 33], name='x')
#x_comp = tf.matmul(tf.reshape(,(-1,33*33)), A1)

A1.shape

for_comp=dataset.map(lambda x:tf.reshape(x,(1,33*33)))
test_x=test_x.map(lambda x:tf.reshape(x,(1,33*33)))



test_x


def mult(x):
    return tf.matmul(x,A1)

#Geetting CS measurements
x_comp=for_comp.map(mult)
test_x=test_x.map(mult)

test_x

train_x=[]
for i in x_comp:
    train_x.append(i)

testing_x=[]
for i in test_x:
  testing_x.append(i)

len(testing_x)

dataset_y=dataset_y.map(lambda x: tf.expand_dims(x,axis=-1))

test_y=test_y.map(lambda x: tf.expand_dims(x,axis=-1))

dataset_y

test_y

len(dataset_y)

test_y

train_y=[]
for i in dataset_y:
    train_y.append(i)

testing_y=[]
for i in test_y:
    testing_y.append(i)

len(testing_y)

# Building model

def my_model():
    inputs=keras.Input(shape=(1,272))
    e1 = tf.keras.layers.Dense(units=1089, activation=tf.nn.relu,
                               )(inputs)
    e1 = tf.reshape(e1,(-1, 33,33,1))
    e2 = tf.keras.layers.Conv2D(64, 11, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))(e1)
    e3 = tf.keras.layers.Conv2D(32, 1, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))(e2)
    e4 = tf.keras.layers.Conv2D(1, 7, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))(e3)
    e5 = tf.keras.layers.Conv2D(64, 11, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))(e4)
    e6 = tf.keras.layers.Conv2D(32, 1, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))(e5)
    e7 = tf.keras.layers.Conv2D(1, 7, padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1))(e6)


    model=keras.Model(inputs=inputs,outputs=e7)

    return model

model_1=my_model()

model_1.summary()

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        return loss
    return contrastive_loss

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.compat.v1.train.exponential_decay(1e-3, global_step,
                                           100000, 0.9, staircase=True)
model_1.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

train_x=tf.convert_to_tensor(train_x)
train_y=tf.convert_to_tensor(train_y)

testing_x=tf.convert_to_tensor(testing_x)
testing_y=tf.convert_to_tensor(testing_y)

print(testing_x.shape)

train_x=tf.reshape(train_x,shape=(27000,1,272))
testing_x=tf.reshape(testing_x,shape=(3000,1,272))

train_x.shape
testing_x.shape

history_1=model_1.fit(train_x, train_y,epochs=150,batch_size=128)

import pandas as pd
# Checkout the history
pd.DataFrame(history_1.history).plot(figsize=(10,7), xlabel="epochs",ylabel="loss");

model_1.evaluate(testing_x, testing_y)

# PSNR

def psnr(img1, img2):
    img1=img1.astype(float)
    img2=img2.astype(float)
    mse=mean_squared_error(img1,img2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# For Gray scale image

path="/content/drive/MyDrive/barbara.bmp"
img=cv2.imread(path)
img= cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
img_lum=img[:,:,0]
plt.imshow(img_lum)

blocks_cam=divide_img_to_blocks(img_lum)
blocks_cam.shape

cs_comp=tf.Variable(blocks_cam,dtype=tf.float32)

cs_comp.shape

cam_comp=tf.reshape(cs_comp,(cs_comp.shape[0],1089))

cam_comp.shape

A1.shape

blocks_cam_test=tf.expand_dims(blocks_cam,axis=-1)
blocks_cam_test.shape

cs_cam=tf.matmul(cam_comp,A1)

cs_cam=tf.reshape(cs_cam,(cs_cam.shape[0],1,272))

cs_cam=tf.convert_to_tensor(cs_cam)

result_cam=model_1.predict(cs_cam)

result_cam.shape

!pip install bm3d
import bm3d

denoise=[]
for i in range(len(result_cam)):
    denoise.append(bm3d.bm3d(result_cam[i],sigma_psd=0.1,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING))

len(denoise)

# Reconstruction Images




# %% RECONSTRUCT IMAGES
test_image = cv2.imread(path)
test_image_lum = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCR_CB)
test_image_lum = test_image_lum[:,:,0]
test_image_lum = cv2.normalize(test_image_lum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
test_gray = cv2.imread(path,0)
filter_size = 33
stride = 14
(h,w) = test_image_lum.shape
h_iters = ((h - filter_size) // stride) + 1
w_iters = ((w - filter_size) // stride) + 1
recon_img = np.zeros((h,w))

tf.math.reduce_max(test_gray)

w_iters

c=-1

for i in range(h_iters):
    for j in range(w_iters):
        #feed = np.expand_dims(test_image_lum[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j],axis=0)
        c=c+1
        out=result_cam[c]
        out = np.squeeze(out)
        recon_img[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j] = out

recon_img

recon_img=recon_img*255.
print(f"PSNR of the  img is  {psnr(recon_img,test_gray)}")

print(f"PSNR of the Reconstruced image and ground truth image without  applying denoise filter is {psnr(recon_img,test_gray)}, for 1000 epochs")
plt.figure(figsize=(12,17))
plt.subplot(2,2,1)
plt.imshow(recon_img,cmap='gray')
plt.axis("off")
plt.subplot(2,2,2)
plt.imshow(test_gray,cmap='gray')
plt.axis("off")


#Training Measurement Matrix


mat=dataset.map(lambda x:tf.reshape(x,(1,33*33)))

train_matrix=[]
for i in mat:
    train_matrix.append(i)

train_matrix=tf.convert_to_tensor(train_matrix)
train_y=tf.convert_to_tensor(train_y)

train_matrix.shape

train_y.shape

model_1.trainable=True

inputs=keras.Input(shape=(1,1089))
y=tf.keras.layers.Dense(units=272,activation=tf.nn.leaky_relu,
                        kernel_initializer=tf.random_normal_initializer)(inputs)
x=model_1(y,training=True)
model_matrix=keras.Model(inputs,outputs=x)

model_matrix.summary()

model_matrix.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=tf.keras.optimizers.Adam(0.001))

history_matrix=model_matrix.fit(train_matrix, train_y,epochs=150,batch_size=64)

import pandas as pd
# Checkout the history
pd.DataFrame(history_matrix.history).plot(figsize=(10,7), xlabel="epochs",ylabel="loss");

cam_comp_mat=tf.reshape(cs_comp,(cs_comp.shape[0],1,1089))

cam_comp_mat=tf.convert_to_tensor(cam_comp_mat)

cam_comp_mat.shape

result_cam_mat=model_matrix.predict(cam_comp_mat)

result_cam_mat.shape

denoise_matrix=[]
for i in range(len(result_cam)):
    denoise_matrix.append(bm3d.bm3d(result_cam_mat[i],sigma_psd=0.1,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING))

len(result_cam_mat)

test_image = cv2.imread(path)
test_image_lum = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCR_CB)
test_image_lum = test_image_lum[:,:,0]
test_image_lum = cv2.normalize(test_image_lum.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
test_gray = cv2.imread(path,0)
filter_size = 33
stride = 14
(h,w) = test_image_lum.shape
h_iters = ((h - filter_size) // stride) + 1
w_iters = ((w - filter_size) // stride) + 1
recon_img_matrix = np.zeros((h,w))

d=-1

for i in range(h_iters):
    for j in range(w_iters):
        #feed = np.expand_dims(test_image_lum[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j],axis=0)
        d=d+1
        out=result_cam_mat[d]
        out = np.squeeze(out)
        recon_img_matrix[stride*i:filter_size+stride*i, stride*j:filter_size+stride*j] = out

d

recon_img_matrix=recon_img_matrix*255.


print(f"PSNR of the Reconstruced image and ground truth after training Measurement Matrix is  {psnr(recon_img_matrix,test_gray)}, for 1000 epochs")
plt.figure(figsize=(12,17))
plt.subplot(2,2,1)
plt.imshow(recon_img_matrix,cmap='gray')
plt.axis("off")
plt.subplot(2,2,2)
plt.imshow(test_gray,cmap='gray')
plt.axis("off")

from skimage.metrics import structural_similarity as ssim
ssim = ssim(recon_img_matrix, test_gray, data_range=255)
print(ssim)
