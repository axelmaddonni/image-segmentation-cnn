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

	x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))( x )
	x = (Conv2D(filter_size, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))( x )
	x = (BatchNormalization())( x )
	x = (Activation('relu'))( x )
	x = (MaxPooling2D((pool_size, pool_size) , data_format=IMAGE_ORDERING  ))( x )
	levels.append( x )

	x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))( x )
	x = (Conv2D(128, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))( x )
	x = (BatchNormalization())( x )
	x = (Activation('relu'))( x )
	x = (MaxPooling2D((pool_size, pool_size) , data_format=IMAGE_ORDERING  ))( x )
	levels.append( x )


	for _ in range(3):
		x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))(x)
		x = (Conv2D(256, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))(x)
		x = (BatchNormalization())(x)
		x = (Activation('relu'))(x)
		x = (MaxPooling2D((pool_size, pool_size) , data_format=IMAGE_ORDERING  ))(x)
		levels.append( x )

	return img_input , levels


def unet( n_classes=4 , l1_skip_conn=True,  input_height=416, input_width=608  ):

	img_input , levels = vanilla_encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4 , f5 ] = levels

	o = f4

	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)

	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

	if l1_skip_conn:
		o = ( concatenate([o,f1],axis=MERGE_AXIS ) )

	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)

	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )

	model = get_segmentation_model(img_input , o )

	return model