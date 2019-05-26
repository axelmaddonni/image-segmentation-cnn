from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate, Reshape, Permute

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model

if IMAGE_ORDERING == 'channels_first':
	MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1

def vanilla_encoder( input_height ,  input_width  ):

	kernel = 3
	filter_size = 64
	pad = 1
	pool_size = 2

	if IMAGE_ORDERING == 'channels_first':
		img_input = Input(shape=(3,input_height,input_width))
	elif IMAGE_ORDERING == 'channels_last':
		img_input = Input(shape=(input_height,input_width , 3 ))

	x = img_input
	levels = []

	for _ in range(4):
		x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))( x )
		x = (Conv2D(filter_size, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))( x )
		x = (BatchNormalization())( x )
		x = (Activation('relu'))( x )
		x = (MaxPooling2D((pool_size, pool_size) , data_format=IMAGE_ORDERING  ))( x )
		levels.append( x )
		filter_size = filter_size * 2

	return img_input , levels


def unet( n_classes=4 , l1_skip_conn=True,  input_height=416, input_width=608  ):

	kernel = 3
	filter_size = 512
	pad = 1
	pool_size = 2

	img_input , levels = vanilla_encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4] = levels

	o = f4

	o = ( ZeroPadding2D( (pad, pad) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (kernel, kernel), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = ( UpSampling2D( (pool_size, pool_size), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )

	o = ( ZeroPadding2D( (pad, pad), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (kernel, kernel), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = ( UpSampling2D( (pool_size, pool_size), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=MERGE_AXIS ) )

	o = ( ZeroPadding2D((pad, pad) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 128, (kernel, kernel), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = ( UpSampling2D( (pool_size, pool_size), data_format=IMAGE_ORDERING))(o)
	if l1_skip_conn: # what is this for ?
		o = ( concatenate([o,f1],axis=MERGE_AXIS ) )

	o = ( ZeroPadding2D((pad, pad)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64, (kernel, kernel), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	### add extra up sampling
	o = ( UpSampling2D( (pool_size, pool_size), data_format=IMAGE_ORDERING))(o)
	###

	o =  Conv2D( n_classes , (kernel, kernel) , padding='same', data_format=IMAGE_ORDERING)(o)

	model = get_segmentation_model(img_input , o )

	return model