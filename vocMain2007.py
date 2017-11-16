import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tfutils import DetectionReader, DetectionMaker
import tensorflow as tf
#image -> (?, 448, 448, 3)
#label -> (?, 7, 7, 25)

IMAGE_SIZE = 448
IMAGE_CHANNEL = 3
CELL_SIZE = 7
rootPath = os.path.join('Total_Data/VOCdevkit2007', 'VOC2007')
AnnotationsPath = os.path.join(rootPath, 'Annotations')
MainPath = os.path.join(rootPath, 'ImageSets', 'Main')
ImagePath = os.path.join(rootPath, 'JPEGImages')
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
TRAIN_RECORDS_COUNT = 5011
TEST_RECORDS_COUNT = 4952

N_CLASSES = len(CLASSES)
PHRASE = 'test'
def readImageFromFile(imageFile):
    imgcv = cv2.imread(imageFile)
    oriH = imgcv.shape[0]
    oriW = imgcv.shape[1]
    imgcv = cv2.resize(imgcv, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return image, oriH, oriW

def readLabelandAnnoFromIndex(xmlFile, oriH, oriW):
    h_ratio = 1.0 * IMAGE_SIZE / oriH
    w_ratio = 1.0 * IMAGE_SIZE / oriW

    label = list()

    tree = ET.parse(xmlFile)
    objs = tree.findall('object')

    CLASS_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))

    for obj in objs:
        bbox = obj.find('bndbox')
        data = dict()
        x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, IMAGE_SIZE - 1), 0)
        y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, IMAGE_SIZE - 1), 0)
        x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, IMAGE_SIZE - 1), 0)
        y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, IMAGE_SIZE - 1), 0)

        cls_ind = CLASS_TO_IND[obj.find('name').text.lower().strip()]

        data['box'] = "%f,%f,%f,%f" % (x1,y1,x2,y2)
        data['classes'] = cls_ind

        label.append(data)
    return label

def ToTFFile():
    if PHRASE == 'train':
        txtname = os.path.join(MainPath, 'trainval.txt')
    else:
        txtname = os.path.join(MainPath, 'test.txt')

    with open(txtname, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]

    TOTAL_COUNT = len(image_index)
    image_list = np.zeros((TOTAL_COUNT, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype=np.uint8)
    label_list = list()
    fileName_list = list()
    # print(len(label_list.shape))
    for i, index in enumerate(image_index):
        imageFile = os.path.join(ImagePath, '%s.jpg' % index)
        image, oriH, oriW = readImageFromFile(imageFile=imageFile)
        annotationFile = os.path.join(AnnotationsPath, '%s.xml' % index)
        label = readLabelandAnnoFromIndex(annotationFile, oriH, oriW)
        image_list[i] = image
        label_list.append(label)
        fileName_list.append(imageFile)
        # "\rImage Loading[%s%s]%d%%"%("*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1))
        s1 = "\rRemain Records Number[%s]" % str(TOTAL_COUNT - i)
        sys.stdout.write(s1)
        sys.stdout.flush()
    maker = DetectionMaker(image_list, label_list, TOTAL_COUNT, '', PHRASE, 2000)
    maker.Make(fileName_list)

    print('Done')

def ToShow():
    fileList = [("%s_%d.tfrecords" % (PHRASE, i)) for i in range(3)]
    print(fileList)
    reader = DetectionReader(fileList, 45)
    list = reader.Read((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))
    filename = list[0]
    image = list[1]
    label = list[2]
    bBox = list[3]
    # xMins = list[3]
    # yMins = list[4]
    # xMaxes = list[5]
    # yMaxes = list[6]
    with tf.Session() as sess:
        tf.train.start_queue_runners()
        filename_batch, image_batch, label_batch, bBox_batch = sess.run([filename, image,label, bBox ])
        image_single = image_batch[0]
        label_list = [int(s) for s in label_batch[0].decode("utf-8").split(',')]
        bBox_single = bBox_batch[0].decode('utf-8')
        bBox_list = bBox_single.split(';')
        bBox = [[float(ss) for ss in s.split(",")] for s in bBox_list]

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(image_single)
        for index, bb in enumerate(bBox):
            rect = mpatches.Rectangle(
                            (bb[0], bb[1]), bb[2]- bb[0], bb[3] - bb[1],fill=False, edgecolor='r', linewidth=3)
            ax.add_patch(rect)
            boxlabel = CLASSES[label_list[index]]
            ax.text(bb[0], bb[1] - 5, boxlabel, family = 'monospace', fontsize = 10)

        # box = [0,0,0,0] #[x1[0], y1[0], x2[0] - x1[0], y2[0] - y1[0]]
        # for i in range(CELL_SIZE):
        #     for j in range(CELL_SIZE):
        #         now_label = label1
        #         if now_label[0] == 1:
        #             boxes = box
        #             rect = mpatches.Rectangle(
        #                 (boxes[0], boxes[1]), boxes[2], boxes[3],fill=False, edgecolor='r', linewidth=3)
        #             ax.add_patch(rect)
        #             boxlabel = CLASSES[np.argmax(now_label[5:25], axis=0)]
        #             ax.text(boxes[0], boxes[1] - 5, boxlabel, family = 'monospace', fontsize = 10)
        plt.show()
if __name__ == '__main__':
    ToTFFile()
    # ToShow()





