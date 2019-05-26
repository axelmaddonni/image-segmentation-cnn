from image_segmentation_cnn.model import unet

from image_segmentation_cnn.data_loader import image_segmentation_generator
import pprint

def main():

    input_height = 1555
    input_width = 1555
    n_classes = 6
    # Class 0 represents UNKNOWN

    images_path = '../data/raw/images'
    annotations_path = '../data/raw/labels_trasposed.csv'
    positions_path = '../data/raw/rowCol.csv'

    generator = image_segmentation_generator(images_path, annotations_path, positions_path, 2, input_height, input_width)
    [img, seg] = next(generator)
    print(img.shape)
    print(seg.shape)

    # model = unet(n_classes=51 ,  input_height=416, input_width=608  )

    # model.train(
    #     train_images =  "../data/raw/images_prepped_train",
    #     train_annotations = "../data/raw/annotations_prepped_train",
    #     epochs=5
    # )

    # out = model.predict_segmentation(
    #     inp="../data/raw/images",
    #     out_fname="out.png"
    # )

    # import matplotlib.pyplot as plt
    # plt.imshow(out)

if __name__ == '__main__':
    main()
