import tensorflow as tf
import os
import numpy as np
from tfutils import TFReader, TFMaker
import matplotlib.pyplot as plt
import pickle
import sys
from PIL import Image
IMAGE_SIZE = 32
IMAGE_CHANNEL = 3
PER_FILE_RECORD_COUNT = 10000

def readDataFromPython(phrase, cifar10or20or100):
    FILEPATH = os.path.join('Total_Data', 'cifar%d' % cifar10or20or100, 'cifar-%d-batches-py' % cifar10or20or100) if cifar10or20or100 == 10 else os.path.join('Total_Data', 'cifar%d' % cifar10or20or100, 'cifar-%d-python' % cifar10or20or100)

    if cifar10or20or100 == 10:
        if phrase == 'train':
            batches = [pickle.load(open(os.path.join(FILEPATH, 'data_batch_%d' % i), 'rb'), encoding='bytes') for i in range(1, 6)]
        else:
            batches = [pickle.load(open(os.path.join(FILEPATH, 'test_batch'), 'rb'), encoding='bytes')]
    else:
        if phrase == 'train':
            batches = [pickle.load(open(os.path.join(FILEPATH, 'train'), 'rb'), encoding='bytes')]
        else:
            batches = [pickle.load(open(os.path.join(FILEPATH, 'test'), 'rb'), encoding='bytes')]
    BATCH_SIZE = 50000 if phrase == 'train' else 10000
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

def ToShow():
    FileList = os.path.join('GenData', 'cifar100')
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
            img=Image.open(curFileName)
            plt.subplot(5,5, idx + 1)
            plt.imshow(img)
        plt.show()
def ToDownloadJpeg():
    PHRASE = ['train', 'test']
    cifar10or20or100 = 100

    ToPath = os.path.join('GenData')
    for phrase in PHRASE:
        GENDATAPATH = os.path.join(ToPath, 'cifar%d' % cifar10or20or100, "%s" % phrase)

        image_list, label_list = readDataFromPython(phrase, cifar10or20or100)
        imageLength = len(image_list)
        imageNameList = []

        for index, imageData in enumerate(image_list):
            label = label_list[index]
            imageName = "image_%d_%d.jpg" % (index, label)
            imageNameList.append(imageName)
            imagePath = os.path.join(GENDATAPATH, imageName)
            plt.imsave(imagePath, imageData)
            s1 = "\r[%d / %d]" % (imageLength - index - 1, imageLength)
            sys.stdout.write(s1)
            sys.stdout.flush()

        maker = TFMaker(imageNameList, batchSize=len(imageNameList), tfFileDir=os.path.join(ToPath, 'cifar%d' % cifar10or20or100,
                                                                                            "%s.tfrecords" % phrase))
        maker.Make()
    # with tarfile.open(os.path.join(ToPath, "cifar%d.tar.gz" % cifar10or20or100),"w:gz") as tar:
    #     tar.add(os.path.join(ToPath, 'cifar%d' % cifar10or20or100), arcname=os.path.basename(os.path.join(ToPath, 'cifar%d' % cifar10or20or100)))


if __name__ == '__main__':
    # ToDownloadJpeg()
    ToShow()




