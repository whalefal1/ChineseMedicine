from keras_segmentation.train import train

if __name__ == '__main__':
    train(
        model_name='resnet50_unet',
        train_images='tongue_data/tongue_data/tongue_data/train_img',
        train_annotations='tongue_data/tongue_data/tongue_data/train_label',
        epochs=30,
        batch_size=4,
        val_images='tongue_data/tongue_data/tongue_data/test_img',
        val_annotations='tongue_data/tongue_data/tongue_data/test_label',
        checkpoints_path='weights/'
    ) 