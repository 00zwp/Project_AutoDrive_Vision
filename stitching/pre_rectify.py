import cv2
import numpy as np
import math


r = 1
search_range = 10
h = 0.05

# 图片矫正
def fun1(image):
    img = cv2.imread(image)
    rows, cols = img.shape[:2]

    # original pts
    pts_o = np.float32([[59,120], [488,52], [144,473], [653,377]])  # 这四个点为原始图片上数独的位置
    pts_d = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])  # 这是变换之后的图上四个点的位置

    # get transform matrix
    M = cv2.getPerspectiveTransform(pts_o, pts_d)
    # apply transformation
    dst = cv2.warpPerspective(img, M, (600, 600))  # 最后一参数是输出dst的尺寸。可以和原来图片尺寸不一致。按需求来确定

    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fun2(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('float32') / 255
    size = img.shape  # h w c

    sr = search_range
    h2 = h * h
    dest = gray.copy()  # python中的变量更像指针，不可直接赋值
    div = -1 * (2 * r + 1) * (2 * r + 1) * h2;

    for y in range(r, size[0] - r):
        print(y)
        for x in range(r, size[1] - r):

            srcblock = gray[y - r:y + r + 1, x - r:x + r + 1]  # 是y,x

            # 限制了搜索范围，不然实在太慢了，做个试验而已
            y_start = max(y - search_range, r)
            x_start = max(x - search_range, r)
            y_end = min(y + search_range, size[0] - r - 1)
            x_end = min(x + search_range, size[1] - r - 1)

            w = np.zeros([y_end - y_start + 1, x_end - x_start + 1])

            for yi in range(y_start, y_end + 1):
                for xi in range(x_start, x_end + 1):
                    # 运动估计简化计算？
                    refblock = gray[yi - r:yi + r + 1, xi - r:xi + r + 1]

                    delta = np.sum(np.square(srcblock - refblock))
                    # print(delta)
                    w[yi - y_start, xi - x_start] = math.exp(delta / div)
                    # print(yi,xi)
                    # time.sleep(1)

            dest[y, x] = np.sum(w * gray[y_start:y_end + 1, x_start:x_end + 1]) / np.sum(w)
    img2=dest.copy()
    img3 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    cv2.imwrite('2.png', img3)
    cv2.imshow('result', img3)
    cv2.waitKey(0)

if __name__ == '__main__':
    # 矫正图片
    img1='111.jpeg'
    fun1(img1)
    # 图片去噪
    img2='222.jpg'
    fun2(img2)