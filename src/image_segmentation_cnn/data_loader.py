import numpy as np
import cv2
import glob
import itertools
import os
from tqdm import tqdm
import random
import pprint

IMAGE_ORDERING = 'channels_first'
# from ..models.config import IMAGE_ORDERING
# from augmentation import augment_seg

random.seed(0)

def get_image_arr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):
	if type( path ) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	if imgNorm == "sub_and_divide":
		img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
	elif imgNorm == "sub_mean":
		img = cv2.resize(img, ( width , height ))
		img = img.astype(np.float32)
		img[:,:,0] -= 103.939
		img[:,:,1] -= 116.779
		img[:,:,2] -= 123.68
	elif imgNorm == "divide":
		img = cv2.resize(img, ( width , height ))
		img = img.astype(np.float32)
		img = img/255.0

	if odering == 'channels_first':
		img = np.rollaxis(img, 2, 0)
	return img

    def read_annotations(path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            annotations = []
            for row in csv_reader:
                annotations.append(row[1:])
            return np.array(annotations)

    def read_annotations_positions(path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            positions = []
            for row in csv_reader:
                positions.append(row[1:])
            return np.array(positions)

    def get_segmentation_arr(image_number, height, width, annotations, positions):
        int_labels = np.zeros(width * height)
        label_number = 0
        for [row, col] in positions:
            int_labels[int(row)*int(col)] = annotations[image_number][label_number]
            label_number+= 1
        return to_categorical(int_labels, num_classes=6)

    def image_segmentation_generator(images_path, annotations_path, positions_path, batch_size, input_height, input_width, do_augment=False ):
        images = glob.glob( os.path.join(images_path,"*.jpg")  ) + glob.glob( os.path.join(images_path,"*.JPG")  )

        image_numbers = list(range(50))
        random.shuffle( image_numbers )
        zipped = itertools.cycle( image_numbers  )

        annotations = read_annotations(annotations_path)
        positions = read_annotations_positions(positions_path)

        while True:
            X = []
            Y = []
            for _ in range(batch_size) :
                image_number = next(zipped)
                img = cv2.imread(images[image_number], cv2.IMREAD_COLOR)

                # if do_augment:
                    # img , seg[:,:,0] = augment_seg( img , seg[:,:,0] )

                X.append(get_image_arr(img , input_width , input_height ,odering=IMAGE_ORDERING))
                Y.append(get_segmentation_arr(image_number, input_height, input_width, annotations, positions))

            pprint.pprint(f"Image shape: {np.array(X).shape} ") # (batch_size, channels, height, width)
            pprint.pprint(f"Annotation shape: {np.array(Y).shape}") # (batch_size, height x width, n_classes)
            yield np.array(X) , np.array(Y)

# Used to create labels_trasposed.csv
def traspose_csv():
    pd.read_csv('../data/raw/labels.csv', header=None).T.to_csv('../data/raw/labels_traposed.csv', header=False, index=False)
