import numpy as np
import matplotlib.pyplot as plt
import os
from tfutils import TFMaker, TFReader
import tensorflow as tf
import sys
from PIL import Image
def ToShow():
    FileList = os.path.join('GenData', 'STL')
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
def ToTfrecords():
    SourceDirPath = os.path.join('Total_Data', 'stl')
    TargetDirPath = os.path.join('GenData', 'STL')
    totalPhrase = ['train', 'test']
    IMAGE_CHANNEL = 3
    IMAGE_SIZE = 96

    for phrase in totalPhrase:
        ImageBinFilePath = os.path.join(SourceDirPath, '%s_X.bin' % phrase)
        LabelBinFilePath = os.path.join(SourceDirPath, '%s_y.bin' % phrase)

        with open(ImageBinFilePath, 'rb') as file:
            everything = np.fromfile(file, dtype=np.uint8)
            images = np.reshape(everything, (-1, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
            image_list = np.transpose(images, (0, 3, 2, 1))

        with open(LabelBinFilePath, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            label_list = np.zeros((len(labels),), dtype=np.int32)
            for index, label in enumerate(labels):
                label_list[index] = int(label)

        TargetImageFilePath = os.path.join(TargetDirPath, phrase)
        fileList = []
        for index, imageData in enumerate(image_list):
            TargetImageFileName = "Image_%d_%d.jpg" % (index, label_list[index])
            fileList.append(TargetImageFileName)
            targetImagePath = os.path.join(TargetImageFilePath, TargetImageFileName)
            plt.imsave(targetImagePath, imageData)
            s1 = "\r[%d / %d]" % (len(image_list) - index - 1, len(image_list))
            sys.stdout.write(s1)
            sys.stdout.flush()

        maker = TFMaker(fileList, batchSize=len(fileList), tfFileDir=os.path.join(TargetDirPath, "%s.tfrecords" % phrase))
        maker.Make()
if __name__ == '__main__':
    # ToTfrecords()
    ToShow()
    # ToUnlabled()