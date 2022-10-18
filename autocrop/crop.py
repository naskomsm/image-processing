import cv2
import os
import glob
import ntpath
from pathlib import Path


def initialize_class_names(source_file):
    with open(source_file, 'rt') as f:
        return f.read().rstrip('\n').split('\n')


def initialize_detection_model():
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net


def get_image_name(imagePath):
    imgName = ntpath.basename(imagePath)
    imgName = imgName.replace(".jpg", "")
    imgName = imgName.replace(".jpeg", "")
    imgName = imgName.replace(".png", "")
    imgName = imgName.replace(".gif", "")
    imgName = imgName.replace(" ", "")
    return imgName


def write_to_file(directory, name, file):
    cv2.imwrite(f"{directory}/{name}.jpeg", file)


classNames = initialize_class_names('coco.names')
net = initialize_detection_model()

path = glob.glob(f"{os.path.join(os.getcwd(), 'images')}/*")

index = 0
for imagePath in path:
    try:
        img = cv2.imread(imagePath)
        imgName = get_image_name(imagePath)

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        directory = os.path.join(os.getcwd(), 'cropped', imgName)
        Path(directory).mkdir(parents=True, exist_ok=True)

        if(len(classIds) > 0 and len(confs) > 0):
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if(classNames[classId-1].upper() == 'BOTTLE' and box[3] >= 700):
                    # box  ->  [X,Y,W,H]
                    cropped_image = img[box[1]:box[1] +
                                        box[3], box[0]:box[0]+box[2]]

                write_to_file(
                    directory, f"{imgName}-{index}", cropped_image)
                print(index)
                index += 1
                

    except Exception as e:
        print(str(e))
