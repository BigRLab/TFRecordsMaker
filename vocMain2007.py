import os
from tfutils import TFReader, TFMaker
TargetDirPath = os.path.join('GenData', 'VOC2007')
PHRASE = ['trainval', 'test']
FILEPATH = os.path.join(TargetDirPath, 'ImageSets', 'Main')
for phrase in PHRASE:
    file = os.path.join(FILEPATH, "%s.txt" % phrase)
    with open(file, 'r') as myfile:
        image_index = [x.strip() for x in myfile.readlines()]

    maker = TFMaker(image_index, batchSize=len(image_index),
                    tfFileDir=os.path.join(TargetDirPath, "%s.tfrecords" % phrase))
    maker.Make()