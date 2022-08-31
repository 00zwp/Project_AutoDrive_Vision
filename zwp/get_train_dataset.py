import math
import os

import matplotlib
import torch
import matplotlib.pyplot as plt
import cv2

def save_train_data(results,index):
    reshape_img = results.imgs[0]
    xyxy = results.xyxy[0].numpy()

    # for i in range(len(xyxy)):
    #     if((xyxy[i][5] == 9)):
    #         light_img = reshape_img[int(xyxy[i][1]):math.ceil(xyxy[i][3]),
    #                     int(xyxy[i][0]):math.ceil(xyxy[i][2])]
    #         light_img = cv2.resize(light_img, (300, 900))
    #         plt.imsave("./zwp/train1/{}.png".format(index), light_img, dpi=600)
    #         index += 1
    # return index

    # 提取到最佳的红绿灯
    best_index = -1
    for i in range(len(xyxy)):
        if ((xyxy[i][5] == 9) and (best_index == -1 or xyxy[i][4] >= xyxy[best_index][4])):
            best_index = i

    # 提取图片中的像素
    if (best_index != -1 and xyxy[best_index][4] > 0):
        light_img = reshape_img[int(xyxy[best_index][1]):math.ceil(xyxy[best_index][3]),
                    int(xyxy[best_index][0]):math.ceil(xyxy[best_index][2])]
        light_img = cv2.resize(light_img, (300, 900))
        plt.imsave("./zwp/train1/{}.png".format(index), light_img, dpi=600)
        index += 1
    return index
def get_trandata():
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    index = 0
    root = "./zwp/save"
    for datapath in os.listdir(root):
        cur_data_path = os.path.join(root, datapath)
        if datapath.find("2022") != -1:
            for img_path in os.listdir(cur_data_path):
                cur_img_path = os.path.join(cur_data_path, img_path)

                results = model(cur_img_path)
                index = save_train_data(results, index)

def delete_file():
    root = "./zwp/train"
    index = 142
    dirlist = os.listdir(root)
    dirlist.sort(key=lambda x:int(x[0:-4]))
    for img_path in dirlist:
        print(img_path)
        if img_path.split(".")[0] == "{}".format(index):
            index += 1
            continue
        else:
            cur_img_path = os.path.join(root, img_path)
            target_img_path = os.path.join(root, "{}.png".format(index))
            os.rename(cur_img_path, target_img_path)
            index+=1

def generate():
    root = "./zwp/train"
    label=0
    with open("./zwp/data.txt", 'a+') as f:
        dirlist = os.listdir(root)
        dirlist.sort(key=lambda x: int(x[0:-4]))
        for img_path in dirlist:
            # label = input("{} label:".format(img_path))
            f.writelines("{} {}\n".format(img_path,label))
        f.close()
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    root = "./train"
    for i in os.listdir(root):
        cur_img_path = os.path.join(root, i)
        img = cv2.imread(cur_img_path)
        img = cv2.resize(img, (100, 300))
        cv2.imwrite(cur_img_path, img)



