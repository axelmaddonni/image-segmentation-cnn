from image_segmentation_cnn.model import unet

def main():
    model = unet(n_classes=51 ,  input_height=416, input_width=608  )

    model.train(
        train_images =  "../data/raw/images",
        train_annotations = "../data/raw/annotations",
        epochs=5
    )

    out = model.predict_segmentation(
        inp="../data/raw/images",
        out_fname="out.png"
    )

    import matplotlib.pyplot as plt
    plt.imshow(out)


if __name__ == '__main__':
    main()
