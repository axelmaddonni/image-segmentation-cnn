from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate, Reshape, Permute, Dropout

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model

import pprint

if IMAGE_ORDERING == 'channels_first':
	MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
	# first layer
	x = Conv2D(
		filters=n_filters,
		kernel_size=(kernel_size, kernel_size),
		kernel_initializer="he_normal",
		padding="same",
		data_format=IMAGE_ORDERING)(input_tensor)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Activation("relu")(x)
	# second layer
	x = Conv2D(
		filters=n_filters,
		kernel_size=(kernel_size, kernel_size),
		kernel_initializer="he_normal",
		padding="same",
		data_format=IMAGE_ORDERING)(x)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Activation("relu")(x)
	return x

def unet(n_classes, input_height, input_width, batchnorm=True):
	n_filters = 64
	kernel = 3
	pool_size = 2
	dropout=0.2
	# pad = 1

	if IMAGE_ORDERING == 'channels_first':
		input_img = Input(shape=(3, input_height, input_width))
	elif IMAGE_ORDERING == 'channels_last':
		input_img = Input(shape=(input_height, input_width, 3 ))

	# contracting path
	c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=kernel, batchnorm=batchnorm)
	p1 = MaxPooling2D(pool_size=pool_size, data_format=IMAGE_ORDERING) (c1)
	p1 = Dropout(dropout)(p1)
	c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=kernel, batchnorm=batchnorm)
	p2 = MaxPooling2D(pool_size=pool_size, data_format=IMAGE_ORDERING) (c2)
	p2 = Dropout(dropout)(p2)
	c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=kernel, batchnorm=batchnorm)
	p3 = MaxPooling2D(pool_size=pool_size, data_format=IMAGE_ORDERING) (c3)
	p3 = Dropout(dropout)(p3)
	c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=kernel, batchnorm=batchnorm)
	p4 = MaxPooling2D(pool_size=pool_size, data_format=IMAGE_ORDERING) (c4)
	p4 = Dropout(dropout)(p4)
	c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=kernel, batchnorm=batchnorm)

	# expansive path
	u6 = Conv2DTranspose(n_filters*8, kernel_size=kernel, strides=pool_size, padding='same', data_format=IMAGE_ORDERING) (c5)
	u6 = concatenate([u6, c4], axis=MERGE_AXIS)
	u6 = Dropout(dropout)(u6)
	c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=kernel, batchnorm=batchnorm)
	u7 = Conv2DTranspose(n_filters*4, kernel_size=kernel, strides=pool_size, padding='same', data_format=IMAGE_ORDERING) (c6)
	u7 = concatenate([u7, c3], axis=MERGE_AXIS)
	u7 = Dropout(dropout)(u7)
	c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=kernel, batchnorm=batchnorm)
	u8 = Conv2DTranspose(n_filters*2, kernel_size=kernel, strides=pool_size, padding='same', data_format=IMAGE_ORDERING) (c7)
	u8 = concatenate([u8, c2], axis=MERGE_AXIS)
	u8 = Dropout(dropout)(u8)
	c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=kernel, batchnorm=batchnorm)
	u9 = Conv2DTranspose(n_filters*1, kernel_size=kernel, strides=pool_size, padding='same', data_format=IMAGE_ORDERING) (c8)
	u9 = concatenate([u9, c1], axis=MERGE_AXIS)
	u9 = Dropout(dropout)(u9)
	c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=kernel, batchnorm=batchnorm)

	output = Conv2D(n_classes, (1, 1), padding='same', data_format=IMAGE_ORDERING, activation='sigmoid')(c9)
	model = get_segmentation_model(input_img, output)
	model.model_name = "unet"
	return model