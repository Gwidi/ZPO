from glob import glob
import os
import random

# update the path if needed
PATH_TO_IMAGES = '../data/coco_val2017/images/'


def generate_train_val_list(path_to_images: str, val_size: float = 0.2, seed: int = 42):
    """This function generates the train and val lists.

    Parameters
    ----------
    path_to_images : str
        The path to the images.
    val_size : float
        The size of the validation set.
    seed : int

    """

    random.seed(seed)
    images_list = sorted(glob(os.path.join(path_to_images, '*.jpg')))
    assert len(images_list) > 0, 'The image list should not be empty, check the PATH_TO_IMAGES variable'

    train_list, val_list = [], []

    # TODO: prepare the train and val lists
    # Note: please do it mannually, do not use sklearn or other libraries
    # 1. shuffle the image list using random.shuffle - note that random.shuffle does not return anything
    # 2. calculate the number of images in the validation set considering the val_size
    # 3. split the image list into train and val lists (train_list and val_list variables)

    # <YOUR CODE HERE>
    random.shuffle(images_list)
    val_count = int(len(images_list) * val_size)
    val_list = images_list[:val_count]
    train_list = images_list[val_count:]


    # TODO: END

    assert len(val_list) > 0, 'The validation set should not be empty'
    assert len(train_list) > 0, 'The training set should not be empty'
    assert len(train_list) + len(val_list) == len(images_list), 'The sum of the training and validation set should be equal to the number of images'
    assert len(train_list) > len(val_list), 'The training set should be larger than the validation set'

    with open('../data/train_list.txt', 'w') as f:
        for image_path in train_list:
            f.write(os.path.abspath(image_path) + '\n')
            
    with open('../data/val_list.txt', 'w') as f:
        for image_path in val_list:
            f.write(os.path.abspath(image_path) + '\n')


if __name__ == '__main__':
    generate_train_val_list(PATH_TO_IMAGES)