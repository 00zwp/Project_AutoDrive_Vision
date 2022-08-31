import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

import torch
import os
from reference_model import CNN

class Object_detector(object):
    def __init__(self):
        # self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.results = None
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to()
        self.reference = CNN(3)
        self.load_references()

    def load_references(self):
        self.reference.load_state_dict(torch.load(r'D:\Project_AutoDrive_Vision\zwp\model.pth'))

    def detect(self, image):

        self.results = self.model(image)

    def show(self):
        img = self.results.imgs[0].astype(np.float)/255.0

        xyxy = self.results.xyxy[0].cpu().numpy()

        for boxes in xyxy:
            if boxes[5] == 9:
                cv2.rectangle(img, (int(boxes[0]),int(boxes[1])), (int(boxes[2]),int(boxes[3])),  color=(255, 255, 255), thickness=5)
                cv2.putText(img, "traffic light", (int(boxes[0]), int(boxes[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("detect", 640, 480);
        cv2.imshow("detect", img)

    def show_traffic_light(self):
        img = self.results.imgs[0].astype(np.float32)/255.0

        xyxy = self.results.xyxy[0].cpu().numpy()

        for boxes in xyxy:
            if boxes[5] == 9:
                light_img = img[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
                print(light_img.shape)
                light_img = cv2.resize(light_img, (100, 300))
                image = torch.from_numpy(light_img.transpose(2, 0, 1)).expand(1, 3, 300, 100)

                prediction = self.reference(image)
                print(prediction.detach().numpy())
                cv2.imshow("light", light_img)
                cv2.waitKey()

if __name__ == '__main__':

    model = Object_detector()

    img = r"D:\Project_AutoDrive_Vision\zwp\save\2022-08-30-17-25-40\for\12480.jpg"
    img = cv2.imread(img)
    model.detect(img[:, 1220:2440])
    model.show_traffic_light()
