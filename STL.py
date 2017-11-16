import tflearn.datasets.mnist as mnist
import numpy as np
import matplotlib.pyplot as plt
import os
from tfutils import ClassficationMaker, ClassficationReader
import tensorflow as tf


PHRASE = 'pred'

FilePath = 'Total_Data/stl'
TRAIN_IMAGE_FILE_PATH = os.path.join(FilePath, 'train_X.bin')
TRAIN_LABEL_FILE_PATH = os.path.join(FilePath, 'train_y.bin')
TEST_IMAGE_FILE_PATH = os.path.join(FilePath, 'test_X.bin')
TEST_LABEL_FILE_PATH = os.path.join(FilePath, 'test_y.bin')
UNLABLED_IMAGE_FILE_PATH = os.path.join(FilePath, 'unlabeled_X.bin')
IMAGE_SIZE = 96
IMAGE_CHANNEL = 3
N_CLASSES = 10

if PHRASE == 'train':
    fileImagePath = TRAIN_IMAGE_FILE_PATH
    fileLabelPath = TRAIN_LABEL_FILE_PATH
    BATCH_SIZE = 5000
elif PHRASE == 'test':
    fileImagePath = TEST_IMAGE_FILE_PATH
    fileLabelPath = TEST_LABEL_FILE_PATH
    BATCH_SIZE = 8000
elif PHRASE == 'pred':
    fileImagePath = UNLABLED_IMAGE_FILE_PATH
    BATCH_SIZE = 5000

def ToShow():
    if PHRASE == 'train':
        filelist = ['train_0.tfrecords']
    elif PHRASE == 'test':
        filelist = ['test_0.tfrecords']
    elif PHRASE == 'pred':
        filelist = ['pred_0.tfrecords']

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
def ToTfrecords():
    with open(fileImagePath, 'rb') as file:
        everything = np.fromfile(file, dtype=np.uint8)
        images = np.reshape(everything, (-1, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        image_list = np.transpose(images, (0, 3, 2, 1))

    if PHRASE != 'pred':
        with open(fileLabelPath, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            label_list = np.zeros((len(labels),), dtype=np.int32)
            for index, label in enumerate(labels):
                label_list[index] = int(label)
    else:
        image_list = image_list[:BATCH_SIZE]
        label_list = np.zeros((BATCH_SIZE,), dtype=np.int32)

    maker = ClassficationMaker(image_list, label_list, BATCH_SIZE, '', PHRASE,perFileRecordCount=BATCH_SIZE)
    maker.Make()
    print('Done')
    # tfm._convertToTFRecords(image_list, label_list, BATCH_SIZE,data_dir='', filename=PHRASE, perfilerecordcount=BATCH_SIZE)

if __name__ == '__main__':
    ToTfrecords()
    # ToShow()
    # ToUnlabled()