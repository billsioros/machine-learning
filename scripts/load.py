from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def get_files(path, extension='jpg'):
    return Path(path).glob(f'*.{extension}')

def get_images(path, extension='jpg'):
    for filename in get_files(path, extension):
        yield filename, Image.open(filename)

def get_resized_images(path, extension='jpg', size=None):
    for filename, image in get_images(path, extension):
        yield filename, image.resize(size) if size is not None else image


def load_images(path, size=(100, 100)):
    images, labels = [], []
    for filename, image in get_resized_images(path, size=size):
        images.append(np.ravel(np.asarray(image)))
        labels.append(filename.name[0])

    if size is None and any(len(_) != len(images[0]) for _ in images):
        max_lenth = len(max(images, key=lambda vector: len(vector)))

        _ = np.zeros((len(images), max_lenth), dtype='uint8')
        for index, image in enumerate(images):
            _[index][0:len(image)] = image

        images = _
    else:
        images = np.array(images, dtype='uint8')

    labels = np.array(LabelEncoder().fit_transform(labels))

    return images, labels


if __name__ == '__main__':
    print(load_images('./img')[0].shape)
