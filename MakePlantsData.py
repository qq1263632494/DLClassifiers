import os
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = '/media/wang/DA18EBFA09C1B27D/数据集/Plants-Seedings-Classification/train/'
tags = ['Black-grass', 'Charlock', 'Cleavers',
        'Common Chickweed', 'Common wheat', 'Fat Hen',
        'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
        'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']


def make_plants_data(IMG_SIZE=128, PATH=''):
    training_data = []
    for i in range(12):
        FILES_DIR = BASE_DIR + tags[i]
        for img in tqdm(os.listdir(FILES_DIR)):
            path = os.path.join(FILES_DIR, img)
            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                training_data.append([np.array(img), i])
            except cv2.error:
                pass
    shuffle(training_data)
    np.save(PATH, training_data)
    print('Data Saved To ' + PATH)
