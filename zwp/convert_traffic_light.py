import cv2
import numpy as np
import matplotlib.pyplot as plt

SEM_COLORS = {
    18: (250, 170, 30),
}

def process_image(image):
    # [H,W,4] --> [H,W,3]
    image = image[:, :, :3]
    # [0,1] --> [0,255]
    image = image * 255
    # Get the r channel
    sem = image[:, :, 0]
    return sem

def visualize_semantic(sem, labels=[18]):
    canvas = np.zeros(sem.shape + (3,), dtype=np.uint8)
    # print("shape of canvas:",canvas.shape)
    for label in labels:
        # print(label)
        canvas[sem == label] = SEM_COLORS[label]
    return canvas

def visualize_iamge_in_semantic(image, sem):
    image = np.asarray(image)
    # [H,W,4] --> [H,W,3]
    image = image[:, :, :3]
    # [0,1] --> [0,255]
    image[sem != 18] = (0,0,0)
    return image

seg_image = plt.imread(r'D:\Project_AutoDrive_Vision\save\2022-08-10-15-25-49\329713.png')
# Get red channel labels
sem = process_image(seg_image)
# Convert to the rgb sematic color
vis_image = visualize_semantic(sem)
plt.imshow(vis_image)
plt.show()

image = plt.imread(r'D:\Project_AutoDrive_Vision\save\2022-08-10-15-25-49\329713-cam.png')
# print(image.shape)
vis_image = visualize_iamge_in_semantic(image, sem)
plt.figure(dpi=1200)
plt.imshow(vis_image)
plt.show()