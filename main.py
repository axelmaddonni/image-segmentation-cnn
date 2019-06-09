from image_segmentation_cnn.model import unet

from image_segmentation_cnn.data_loader import image_segmentation_generator
import pprint

def main():

    # input_height = 1555
    input_height = 1536

    # input_width = 1555
    input_width = 1536
    n_classes = 6
    # Class 0 represents UNKNOWN

    images_path = 'data/raw/images'
    annotations_path = 'data/raw/labels_trasposed.csv'
    positions_path = 'data/raw/rowCol.csv'
    checkpoints_path = 'models/checkpoint'

    model = unet(n_classes=n_classes, input_height=input_height, input_width=input_width)

    model.summary()

    model.train(
        train_images = images_path,
        train_annotations = annotations_path,
        train_positions = positions_path,
        checkpoints_path = checkpoints_path,
        epochs=2,
        batch_size=1
    )

    # out = model.predict_segmentation(
    #     inp="../data/raw/images",
    #     out_fname="out.png"
    # )

    # import matplotlib.pyplot as plt
    # plt.imshow(out)

if __name__ == '__main__':
    main()
