import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

import torch
import os
from detection.reference_model import CNN

class Object_detector(object):
    def __init__(self):
        # self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.labels = ["red light", "yellow light", "green light"]
        self.results = None
        self.image = None
        print("加载目标检测模型")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        print("加载红绿灯识别模型")
        self.reference = CNN(3)
        self.load_references()

    def load_references(self):
        self.reference.load_state_dict(torch.load('./detection/model.pth'))

    def detect(self, image):
        # image = cv2.resize(image, (1280, 2440))
        if image.max() > 1.0:
            self.image = image.astype(np.float32)/255.0
        else:
            self.image = image
        self.results = self.model(image)

    def show(self):
        self.results.render()
        detect = self.results.imgs[0]

        cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("detect", 640, 480)
        cv2.imshow("detect", detect)
        # cv2.waitKey()

    def show_traffic_light(self):
        best_conv = np.zeros(3)
        best_image = None
        xyxy = self.results.xyxy[0].cpu().numpy()

        best = 0
        for index, boxes in enumerate(xyxy):
            if boxes[5] == 9:
                if boxes[4] >= xyxy[best][4]:
                    best = index
                    best_image = self.image[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
                # light_img = self.image[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
                # light_img = cv2.resize(light_img, (40, 120))
                # image = torch.from_numpy(light_img.transpose(2, 0, 1)).expand(1, 3, 120, 40)
                # prediction = self.reference(image).detach().numpy()
                # if best_conv.max() < prediction.max():
                #     best_conv = prediction
                #     best_image = light_img

        cv2.namedWindow("light", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("light", 100, 300)
        if best_image is not None:
            light_img = cv2.resize(best_image, (40, 120))
            image = torch.from_numpy(light_img.transpose(2, 0, 1)).expand(1, 3, 120, 40)
            prediction = self.reference(image).detach().numpy()
            print(self.labels[np.argmax(prediction)])
            best_image = cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("light", best_image)
            # cv2.waitKey()
        else:
            print("no ligjts")
            best_image = cv2.imread("./Control/traffic_light/4.png")
            cv2.imshow("light", best_image)

if __name__ == '__main__':

    model = Object_detector()
    img = r"D:\Project_AutoDrive_Vision\zwp\save\2022-08-30-17-25-40\for\12480.jpg"
    img = cv2.imread(img)
    model.detect(img)
    model.show_traffic_light()
