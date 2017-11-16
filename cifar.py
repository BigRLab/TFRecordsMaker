import tensorflow as tf
import os
import numpy as np
from tfutils import ClassficationMaker, ClassficationReader
import matplotlib.pyplot as plt
import pickle
from PIL import Image

PHRASE = 'test'
cifar10or20or100 = 100
FILEPATH = os.path.join('Total_Data', 'cifar%d' % cifar10or20or100, 'cifar-%d-batches-py' % cifar10or20or100) if cifar10or20or100 == 10 else os.path.join('Total_Data', 'cifar%d' % cifar10or20or100, 'cifar-%d-python' % cifar10or20or100)
IMAGE_SIZE = 32
IMAGE_CHANNEL = 3
PER_FILE_RECORD_COUNT = 10000
BATCH_SIZE = 50000 if PHRASE == 'train' else 10000
N_CLASSES = cifar10or20or100

TFFILEPATH = ''
TFFILE_TRAIN_FILE_LIST = [os.path.join(TFFILEPATH, 'train_%d.tfrecords' % i) for i in range(5)]
TFFILE_TTEST_FILE_LIST = [os.path.join(TFFILEPATH, 'test_%d.tfrecords' % i) for i in range(1)]

def readDataFromPython():
    if cifar10or20or100 == 10:
        if PHRASE == 'train':
            batches = [pickle.load(open(os.path.join(FILEPATH, 'data_batch_%d' % i), 'rb'), encoding='bytes') for i in range(1, 6)]
        else:
            batches = [pickle.load(open(os.path.join(FILEPATH, 'test_batch'), 'rb'), encoding='bytes')]
    else:
        if PHRASE == 'train':
            batches = [pickle.load(open(os.path.join(FILEPATH, 'train'), 'rb'), encoding='bytes')]
        else:
            batches = [pickle.load(open(os.path.join(FILEPATH, 'test'), 'rb'), encoding='bytes')]

    images = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype=np.uint8)
    labels = np.zeros((BATCH_SIZE,), dtype=np.int32)
    for i, b in enumerate(batches):
        if cifar10or20or100 == 10:
            for j, l in enumerate(b[b'labels']):
                images[i*PER_FILE_RECORD_COUNT + j] = b[b'data'][j].reshape([IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*PER_FILE_RECORD_COUNT + j] = l
        elif cifar10or20or100 == 20:
            for j, l in enumerate(b[b'coarse_labels']):
                images[i*PER_FILE_RECORD_COUNT + j] = b[b'data'][j].reshape([IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*PER_FILE_RECORD_COUNT + j] = l
        else:
            for j, l in enumerate(b[b'fine_labels']):
                images[i*PER_FILE_RECORD_COUNT + j] = b[b'data'][j].reshape([IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*PER_FILE_RECORD_COUNT + j] = l
    return images, labels

def ToTFRecords():
    image_list, label_list = readDataFromPython()
    maker = ClassficationMaker(image_list, label_list, BATCH_SIZE, '', PHRASE, PER_FILE_RECORD_COUNT)
    maker.Make()
    print('Done')

def ToShow():
    if PHRASE == 'train':
        filelist = TFFILE_TRAIN_FILE_LIST
    else:
        filelist = TFFILE_TTEST_FILE_LIST

    reader = ClassficationReader(filelist, BATCH_SIZE)
    list = reader.Read((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))
    images = list[0]
    labels = list[1]

    with tf.Session() as sess:
        tf.train.start_queue_runners()
        realImages, realLabels = sess.run([images, labels])
        for idx in range(6):
            plt.subplot(2,3, idx + 1)
            plt.imshow(realImages[idx])
            plt.title(realLabels[idx])
        plt.show()
if __name__ == '__main__':
    # ToTFRecords()
    ToShow()

