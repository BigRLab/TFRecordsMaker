import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tfRecordsMaker
import tfRecordsReader
import tensorflow as tf
#image -> (?, 448, 448, 3)
#label -> (?, 7, 7, 25)

IMAGE_SIZE = 448
IMAGE_CHANNEL = 3
CELL_SIZE = 7
rootPath = os.path.join('VOCdevkit', 'VOC2007')
AnnotationsPath = os.path.join(rootPath, 'Annotations')
MainPath = os.path.join(rootPath, 'ImageSets', 'Main')
ImagePath = os.path.join(rootPath, 'JPEGImages')
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
TRAIN_RECORDS_COUNT = 5011
TEST_RECORDS_COUNT = 4952


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

    label = np.zeros((CELL_SIZE, CELL_SIZE, 25))
    tree = ET.parse(xmlFile)
    objs = tree.findall('object')

    CLASS_TO_IND = dict(zip(CLASSES, range(len(CLASSES))))

    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, IMAGE_SIZE - 1), 0)
        y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, IMAGE_SIZE - 1), 0)
        x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, IMAGE_SIZE - 1), 0)
        y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, IMAGE_SIZE - 1), 0)

        cls_ind = CLASS_TO_IND[obj.find('name').text.lower().strip()]
        boxes = [x1, y1, x2 - x1, y2 - y1]
        x_ind = int(boxes[0] * CELL_SIZE / IMAGE_SIZE)
        y_ind = int(boxes[1] * CELL_SIZE / IMAGE_SIZE)
        if label[y_ind, x_ind, 0] == 1:
            continue
        label[y_ind, x_ind, 0] = 1
        label[y_ind, x_ind, 1:5] = boxes
        label[y_ind, x_ind, 5 + cls_ind] = 1
    return label, len(objs)

if __name__ == '__main__':
    # if PHRASE == 'train':
    #     txtname = os.path.join(MainPath, 'trainval.txt')
    # else:
    #     txtname = os.path.join(MainPath, 'test.txt')
    #
    # with open(txtname, 'r') as f:
    #     image_index = [x.strip() for x in f.readlines()]
    #
    # TOTAL_COUNT = len(image_index)
    # image_list = np.zeros((TOTAL_COUNT, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype=np.uint8)
    # label_list = np.zeros((TOTAL_COUNT, CELL_SIZE, CELL_SIZE, 25), dtype=np.float32)
    # print(len(label_list.shape))
    # for i, index in enumerate(image_index):
    #     imageFile = os.path.join(ImagePath, '%s.jpg' % index)
    #     image, oriH, oriW = readImageFromFile(imageFile=imageFile)
    #     annotationFile = os.path.join(AnnotationsPath, '%s.xml' % index)
    #     label, length = readLabelandAnnoFromIndex(annotationFile, oriH, oriW)
    #     image_list[i] = image
    #     label_list[i] = label
    #     # "\rImage Loading[%s%s]%d%%"%("*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1))
    #     s1 = "\rRemain Records Number[%s]" % str(TOTAL_COUNT - i)
    #     sys.stdout.write(s1)
    #     sys.stdout.flush()
    #
    # tfRecordsMaker._convertToTFRecords(images=image_list,
    #                                    labels=label_list, batch_size=TOTAL_COUNT,
    #                                    data_dir='', filename=PHRASE,
    #                                    perfilerecordcount=1000)
    # print('Done')

    fileList = [("%s_%d.tfrecords" % (PHRASE, i)) for i in range(1)]
    print(fileList)
    image, label = \
        tfRecordsReader.readFromTFRecords(fileList, batch_size=1,
                                          img_shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], IsLabelInt=False)

    plt.figure(figsize=(10,10), facecolor='w')
    with tf.Session() as sess:
        tf.train.start_queue_runners()
        image_batch, label_batch = sess.run([image, label])
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(image_batch[0])
        label1 = label_batch[0]
        for i in range(CELL_SIZE):
            for j in range(CELL_SIZE):
                now_label = label1[i][j]
                if now_label[0] == 1:
                    boxes = now_label[1:5]
                    rect = mpatches.Rectangle(
                        (boxes[0], boxes[1]), boxes[2], boxes[3],fill=False, edgecolor='r', linewidth=3)
                    ax.add_patch(rect)
                    boxlabel = CLASSES[np.argmax(now_label[5:25], axis=0)]
                    ax.text(boxes[0], boxes[1] - 5, boxlabel, family = 'monospace', fontsize = 10)
        plt.show()




    # image1 = image_list[1123]
    # label1 = label_list[1123]
    # print(image1.shape)
    # print(label1)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(image1)
    #
    # for i in range(CELL_SIZE):
    #     for j in range(CELL_SIZE):
    #         now_label = label1[i][j]
    #         if now_label[0] == 1:
    #             boxes = now_label[1:5]
    #             rect = mpatches.Rectangle(
    #                 (boxes[0], boxes[1]), boxes[2], boxes[3],fill=False, edgecolor='r', linewidth=3)
    #             ax.add_patch(rect)
    #             boxlabel = CLASSES[np.argmax(now_label[5:25], axis=0)]
    #             ax.text(boxes[0], boxes[1] - 5, boxlabel, family = 'monospace', fontsize = 10)
    # plt.show()