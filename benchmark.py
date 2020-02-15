#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import cv2
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from testYolo import *

test_path = "../coco/val2014"
save_path = "../coco/val2014/predictions"

#test_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

test_classes = ['tello']

#topk
#topk = 1
topk = 100


images_folder = os.path.join(test_path, "images")
annotations_folder = os.path.join(test_path, "annotations")

images = os.listdir(images_folder)

def detect(image_name):
    prediction_path = os.path.join(save_path, image_name)
    b = None
    box = []
    with open(prediction_path, 'r') as f:
        prediction_raw = f.read()
        for prBox in prediction_raw.split('\n'):

            pred = prBox.split(" ")
            if pred == [""]:
                continue
            #print(pred)
            try: 
                class_name, score, x, y, w, h = pred #pred[0], pred[1], pred[2], pred[3], pred[4]
            except IndexError:
                continue
            score = float(score)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            b = (class_name, score, (x, y, w, h))
            box.append(b)


    return box
    '''
    with open(prediction_path, 'r') as f:
        prediction_raw = f.read()
        pred = prediction_raw.(" ")
    '''

def sign(x):
    if x>=0: return 1
    else: return -1

def get_IoU(boxA, boxB):
    classA, confidenceA, positionA = boxA
    classB, confidenceB, positionB = boxB
    if classA != classB:
        return 0

    xA, yA, wA, hA = positionA
    xB, yB, wB, hB = positionB
    
    tx = sign(xB - xA)
    ty = sign(yB - yA)

    Ix = (wA/2 + wB/2 + tx*(xA - xB))
    Iy = (hA/2 + hB/2 + ty*(yA - yB))

    if Ix < 0 or Iy < 0:
        I = 0
    else:
        I = Ix * Iy

    U = wA*hA + wB*hB - I
    return I/U

def main():
    TPs = np.zeros(topk)
    ABs = np.zeros(topk)
    PBs = np.zeros(topk)

    precision = np.empty(0)
    recall = np.empty(0)
    ioulist = np.array([1, 0])

    n = 0
    for image_name in images:
        n += 1
        print("{} Image {}".format(n, image_name))
        image_path = os.path.join(images_folder, image_name)
        annotation_path = os.path.join(annotations_folder, image_name[:-4] + ".txt")
        if not os.path.exists(annotation_path):
            continue

        img = cv2.imread(image_path)

        with open(annotation_path, "r") as f:
            annotation_raw = f.read()
        annotation = []
        for box_raw in annotation_raw.split("\n"):
            pack = box_raw.split(" ")
            try: 
                class_name, score, x, y, w, h = pack[0], pack[1], pack[2], pack[3], pack[4], pack[5]
            except IndexError:
                continue
            class_name = test_classes[int(class_name)]
            #img.shape[0] : height
            #img.shape[1] : width
            x = img.shape[1]*float(x)
            y = img.shape[0]*float(y)
            w = img.shape[1]*float(w)
            h = img.shape[0]*float(h)
            box = (class_name, score, (x, y, w, h))
            annotation.append(box)

        prediction = detect(image_name[:-4] + ".txt")
        if len(prediction) == 0:
            continue

        #print("predicted {} over {} images".format(len(prediction), len(annotation)))

        TP = np.zeros(topk)

        for k in range(topk):
            if k < len(prediction):
                IoU_list = []
                for box in annotation:
                    pred_box = prediction[k]
                    IoU = get_IoU(box, pred_box)
                    #print("IOU ", IoU)
                    IoU_list.append(IoU)
                if max(IoU_list) > 0.5:
                    TP[k] = 1 + TP[k-1]
                else:
                    TP[k] = 0 + TP[k-1]
            else:
                TP[k] = TP[len(prediction)-1]

        PB = np.zeros(topk)
        if topk <= len(prediction):
            for k in range(topk):
                PB[k] = k+1
        else:
            for k in range(len(prediction)):
                PB[k] = k+1
            for k in range(len(prediction), topk):
                PB[k] = len(prediction)

        AB = np.zeros(topk)
        AB += len(annotation)

                    #k top k predictions
        TPs += TP   #true positive
        PBs += PB   #predicted boxes
        ABs += AB   #annotated boxes
        #print("TP {}\nPB {}\nAB {}".format(TP, PB, AB))
        #import pdb
        #pdb.set_trace()
        #print("TPs {}\nPBs {}\nABs{}".format(TPs, PBs, ABs))

        #p = TPs/PBs
        #r = TPs/ABs
        #hey = np.hstack((r, p))

        #ioulist = np.vstack((ioulist, hey))

        precision = TPs/PBs
        recall = TPs/ABs

        #print("precision    {}".format(precision))
        #print( "recall      {}".format(recall))

        """
        if n % 5 == 0:
            ioulist = np.vstack((ioulist, hey))
            TPs = np.zeros(topk)
            ABs = np.zeros(topk)
            PBs = np.zeros(topk)
        """

             

    recall 		= np.hstack((0,recall,1))
    precision 	= np.hstack((1, precision, 0))

    for i in range(len(recall)):
        print(recall[i], precision[i])
    """
    print("ioulist")
    ioulist = ioulist[np.argsort(ioulist[:, 1])[::-1]]
    ioulist = np.vstack(([0, 1], ioulist))
    
    recall = ioulist[:, 0]
    precision = ioulist[:,1]
    print(ioulist)
    """

    mAP = 0
    for i in range(len(recall) -1):
        dw = recall[i+1] - recall[i]
        h = (precision[i+1] + precision[i])/2
        mAP += h*dw

    print("mAP ", mAP)
    plt.plot(recall, precision, "ro")

    plt.show()

if __name__ == "__main__":
    main()