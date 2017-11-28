import os
import cv2
import numpy as np
from tfutils import TFReader, TFMaker
import tensorflow as tf
import matplotlib.pyplot as plt
from tflearn.datasets import oxflower17
import shutil

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
    image = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return image

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
    tarFolder = os.path.join('GenData', 'flowers')
    trainTarFolder = os.path.join(tarFolder, 'train')
    TrainFileList = []
    for file in trainFileList:
        sourceFilePath = file[0]
        label = file[1]
        temp = sourceFilePath.split('/')
        sourceFileName = temp[len(temp) - 1]
        targetFileName = sourceFileName.split('.')[0] + '_' + label + "." + sourceFileName.split('.')[1]
        targetFilePath = os.path.join(trainTarFolder, targetFileName)
        shutil.copy(sourceFilePath, targetFilePath)
        TrainFileList.append(targetFilePath)
    maker = TFMaker(TrainFileList, batchSize=len(TrainFileList),
                    tfFileDir=os.path.join(tarFolder, "train.tfrecords"))
    maker.Make()
    testTarFolder = os.path.join(tarFolder, 'test')
    TestFileList = []
    for file in trainFileList:
        sourceFilePath = file[0]
        label = file[1]
        temp = sourceFilePath.split('/')
        sourceFileName = temp[len(temp) - 1]
        targetFileName = sourceFileName.split('.')[0] + '_' + label + "." + sourceFileName.split('.')[1]
        targetFilePath = os.path.join(testTarFolder, targetFileName)
        shutil.copy(sourceFilePath, targetFilePath)
        TestFileList.append(targetFilePath)
    maker = TFMaker(TestFileList, batchSize=len(TestFileList),
                    tfFileDir=os.path.join(tarFolder, "test.tfrecords"))
    maker.Make()


def ToShow():
    FileList = os.path.join('GenData', 'flowers')
    TrainFile = [os.path.join(FileList, 'train.tfrecords')]
    TestFile = os.path.join(FileList, 'test.tfrecords')

    reader = TFReader(TrainFile, batch_size=10)
    fileList = reader.Read()

    with tf.Session() as sess:
        tf.train.start_queue_runners()
        realFileName = sess.run(fileList)
        for idx in range(10):
            curFileName = realFileName[idx]
            print(curFileName)
            curFileName = os.path.join(FileList, 'train', curFileName.decode())
            img=readImageFromFile(curFileName)
            plt.subplot(5,5, idx + 1)
            plt.imshow(img)
        plt.show()
if __name__ == '__main__':
    # download()
    # ToTfrecords()
    ToShow()