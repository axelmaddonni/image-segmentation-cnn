import argparse
import json
import os
import six

from .data_loader import image_segmentation_generator

def find_latest_checkpoint( checkpoints_path ):
	ep = 0
	r = None
	while True:
		if os.path.isfile( checkpoints_path + "." + str( ep )  ):
			r = checkpoints_path + "." + str( ep )
		else:
			return r

		ep += 1

def train( model,
		train_images,
		train_annotations,
		train_positions,
		input_height=None,
		input_width=None,
		n_classes=None,
		verify_dataset=True,
		checkpoints_path=None,
		epochs = 5,
		batch_size = 2,
		validate=False,
		val_images=None,
		val_annotations=None,
		val_positions=None,
		val_batch_size=2,
		auto_resume_checkpoint=False,
		load_weights=None,
		steps_per_epoch=512,
		optimizer_name='adadelta'
	):

	from . import model_from_name

	if  isinstance(model, six.string_types) : # check if user gives model name insteead of the model object
		# create the model from the name
		assert ( not n_classes is None ) , "Please provide the n_classes"
		if (not input_height is None ) and ( not input_width is None):
			model = model_from_name[ model ](  n_classes , input_height=input_height , input_width=input_width )
		else:
			model = model_from_name[ model ](  n_classes )

	n_classes = model.n_classes
	input_height = model.input_height
	input_width = model.input_width
	output_height = model.output_height
	output_width = model.output_width

	if validate:
		assert not ( val_images is None )
		assert not ( val_annotations is None )

	if not optimizer_name is None:
		model.compile(loss='categorical_crossentropy',
			optimizer= optimizer_name ,
			metrics=['accuracy'])

	if not checkpoints_path is None:
		open( checkpoints_path+"_config.json" , "w" ).write( json.dumps( {
			"model_class" : model.model_name ,
			"n_classes" : n_classes ,
			"input_height" : input_height ,
			"input_width" : input_width ,
			"output_height" : output_height ,
			"output_width" : output_width
		}))

	if ( not (load_weights is None )) and  len( load_weights ) > 0:
		print("Loading weights from " , load_weights )
		model.load_weights(load_weights)

	if auto_resume_checkpoint and ( not checkpoints_path is None ):
		latest_checkpoint = find_latest_checkpoint( checkpoints_path )
		if not latest_checkpoint is None:
			print("Loading the weights from latest checkpoint ",latest_checkpoint )
			model.load_weights( latest_checkpoint )

	train_gen = image_segmentation_generator(train_images, train_annotations, train_positions, batch_size, input_height, input_width)

	if validate:
		val_gen  = image_segmentation_generator(val_images, val_annotations, val_positions, val_batch_size, input_height , input_width)

	class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

	if not validate:
		for ep in range( epochs ):
			print("Starting Epoch " , ep )
			model.fit_generator( train_gen, steps_per_epoch, epochs=1, class_weight=class_weights)
			if not checkpoints_path is None:
				model.save_weights( checkpoints_path + "." + str( ep ) )
				print("saved ", checkpoints_path + ".model." + str( ep ) )
			print("Finished Epoch" , ep )
	else:
		for ep in range( epochs ):
			print("Starting Epoch " , ep )
			model.fit_generator( train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=200, epochs=1, class_weight=class_weights)
			if not checkpoints_path is None:
				model.save_weights( checkpoints_path + "." + str( ep ))
				print("saved ", checkpoints_path + ".model." + str( ep ))
			print("Finished Epoch" , ep )
