import os
import pickle
import cv2
import numpy as np
import random
from tfutils import ClassficationReader, ClassficationMaker
import tensorflow as tf
import matplotlib.pyplot as plt
from tflearn.datasets import oxflower17

IMAGE_SIZE = 224
IMAGE_CHANNEL = 3
FILEPATH = 'Total_Data/flowers17/jpg'
N_CLASSES = 17
TRAIN_COUNT = 70
TEST_COUNT = 80-TRAIN_COUNT

def download():
    oxflower17.load_data('Total_Data/flowers17/')

def readImageFromFile(imageFile):
    imgcv = cv2.imread(imageFile)
    oriH = imgcv.shape[0]
    oriW = imgcv.shape[1]
    imgcv = cv2.resize(imgcv, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return image, oriH, oriW

def ToTfrecords():
    fileList = list()
    g = os.walk(FILEPATH)
    trainFileList = []
    testFileList = []
    index = 0
    for path,d,filelist in g:
        for directory in d:
            path = os.path.join(FILEPATH, directory)
            p = os.walk(path)
            index = 0
            for path , d, filelist in p:
                for filename in filelist:
                    if filename.endswith('jpg'):
                        index = index + 1
                        if index <= TRAIN_COUNT:
                            trainFileList.append([os.path.join(path, filename), directory])
                        else:
                            testFileList.append([os.path.join(path, filename), directory])

    random.shuffle(trainFileList)
    random.shuffle(testFileList)

    image_list = np.zeros((1190, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype=np.uint8)
    label_list = np.zeros((1190,), dtype=np.int32)

    for index, file in enumerate(trainFileList):
        filename = file[0]
        filelabel = int(file[1])
        image,_,_ = readImageFromFile(filename)
        image_list[index] = image
        label_list[index] = filelabel

    maker = ClassficationMaker(image_list, label_list,  1190, '', 'train', perFileRecordCount=1190)
    maker.Make()

    image_list = np.zeros((170, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype=np.uint8)
    label_list = np.zeros((170,), dtype=np.int32)
    for index, file in enumerate(testFileList):
        filename = file[0]
        filelabel = int(file[1])
        image,_,_ = readImageFromFile(filename)
        image_list[index] = image
        label_list[index] = filelabel

    maker = ClassficationMaker(image_list, label_list,  170, '', 'test', perFileRecordCount=170)
    maker.Make()

    print('Done')
def ToShow():
    reader = ClassficationReader(['train_0.tfrecords'], 1190)
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
    # download()
    # ToTfrecords()
    ToShow()