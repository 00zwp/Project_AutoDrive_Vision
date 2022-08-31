import glob
import logging
import os
import sys
import datetime
from queue import Queue, Empty
import matplotlib.pyplot as plt

savepath = './save/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.makedirs(savepath, exist_ok=True)
global img_index
img_index = 0
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import cv2

IM_WIDTH = 590
IM_HEIGHT = 1640

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def calc_slope(line):
    x1, y1, x2, y2 = line[0]
    return (y2 - y1) / (x2 - x1)


def line_selection(lines, threshold):
    slopes = [calc_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    # return lines


def least_squres_fit(lines):
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    poly = np.polyfit(x_coords, y_coords, deg=1)
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)

def getLane(filename,savename1,savename2):
    img_path = filename

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    low_threshold = 150
    high_threshold = 300

    canny_img = cv2.Canny(gray, low_threshold, high_threshold)

    # cut interested img
    left_bottom = [0, canny_img.shape[0]]
    right_bottom = [canny_img.shape[1] - 60, canny_img.shape[0]]
    # left_bottom = [400, 600]
    # right_bottom = [1000, 600]
    # apex = [800, 300]
    apex = [canny_img.shape[1] / 2, 300]
    vertices = np.array([left_bottom, right_bottom, apex], np.int32)
    roi_img = region_of_interest(canny_img, vertices)

    # hough lines operation
    lines = cv2.HoughLinesP(roi_img, 1, theta=np.pi / 180, threshold=15, minLineLength=10, maxLineGap=20)

    # classify the line by slope
    left_lines = [line for line in lines if calc_slope(line) > 0]
    right_lines = [line for line in lines if calc_slope(line) < 0]

    # use threshold to modify the lines to select lines.
    print("before operation: {}".format(len(left_lines)))
    line_selection(left_lines, threshold=0.1)
    line_selection(right_lines, threshold=0.1)
    print("after operation: {}".format(len(left_lines)))

    left_line = least_squres_fit(left_lines)
    right_line = least_squres_fit(right_lines)

    blackImg = np.zeros_like(img)

    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=[255, 0, 0], thickness=6)
    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=[255, 0, 0], thickness=6)

    cv2.line(blackImg, tuple(left_line[0]), tuple(left_line[1]), color=[255, 0, 0], thickness=6)
    cv2.line(blackImg, tuple(right_line[0]), tuple(right_line[1]), color=[255, 0, 0], thickness=6)

    # cv2.imshow('gray', blackImg)
    # cv2.waitKey(0)

    plt.imsave(savename1,img)
    plt.imsave(savename2,blackImg)

def process_image(image):
    # [H,W,4] --> [H,W,3]
    image = image[:, :, :3]

    # [0,1] --> [0,255]
    image = image * 255

    # Get the r channel
    sem = image[:, :, 0]

    return sem

SEM_COLORS = {

    6: (157, 234, 50),

}

def visualize_semantic(sem, labels=[ 6]):
    canvas = np.zeros(sem.shape + (3,), dtype=np.uint8)
    # print("shape of canvas:",canvas.shape)
    for label in labels:
        # print(label)
        canvas[sem == label] = SEM_COLORS[label]

    return canvas

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    if 'lidar' in sensor_name:
        sensor_data.save_to_disk(os.path.join(savepath, '%06d.ply' % sensor_data.frame))
    if 'camera' in sensor_name:
        sensor_data.save_to_disk(os.path.join(savepath, '%06d.png' % sensor_data.frame))
    if 'segmentation' in sensor_name:
        sensor_data.save_to_disk(os.path.join(savepath, '%06d.png' % sensor_data.frame))#, carla.ColorConverter.CityScapesPalette)
        sensor_data.save_to_disk(os.path.join(savepath, '%06d-all.png' % sensor_data.frame), carla.ColorConverter.CityScapesPalette)

        #提取车道线
        filename=os.path.join(savepath, '%06d.png' % sensor_data.frame)
        image = plt.imread(filename)
        sem = process_image(image)
        # Convert to the rgb sematic color
        vis_image = visualize_semantic(sem)
        plt.imsave(os.path.join(savepath, '%06d-lane.png' % sensor_data.frame),vis_image)

        try:
            getLane(os.path.join(savepath, '%06d-lane.png' % sensor_data.frame),os.path.join(savepath, '%06d-lane-img-line.png' % sensor_data.frame),os.path.join(savepath, '%06d-lane-line.png' % sensor_data.frame))
        except:
            pass
    if 'laneDetector' in sensor_name:
        print(sensor_data)
    sensor_queue.put((sensor_data.frame, sensor_name))


actor_list = []
try:
    # 获取客户
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    # 获取世界
    world = client.get_world()

    # 将仿真世界设置为异步状态
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    sensor_queue = Queue()

    # ——————————————————————————————————加载汽车——————————————————————————
    # 获取蓝图
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    # 随机选择出生点
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    # ——————————————————————————————————加载语义分割器——————————————————————————
    # 加载传感器蓝图设置
    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint_forward = blueprint_library.find('sensor.camera.semantic_segmentation')
    # change the dimensions of the image
    blueprint_forward.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint_forward.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint_forward.set_attribute('fov', '60')
    # Set the time in seconds between sensor captures
    blueprint_forward.set_attribute('sensor_tick', '0.5')
    # Adjust sensor relative to vehicle
    spawn_point_seg = carla.Transform(carla.Location(x=0, y=0, z=10), carla.Rotation(pitch=-45))
    # spawn the sensor and attach to vehicle.
    sensor_seg = world.spawn_actor(blueprint_forward, spawn_point_seg, attach_to=vehicle,
                               attachment_type=carla.AttachmentType.Rigid)
    sensor_seg.listen(lambda data: sensor_callback(sensor_data=data, sensor_queue=sensor_queue, sensor_name='segmentation'))
    actor_list.append(sensor_seg)

    #——————————————————————————————————加载前置传感器——————————————————————————
    blueprint_forward = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint_forward.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint_forward.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint_forward.set_attribute('fov', '60')
    # Set the time in seconds between sensor captures
    blueprint_forward.set_attribute('sensor_tick', '0.5')
    # Adjust sensor relative to vehicle
    spawn_point_forward = (carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(0,0,0)))
    # spawn the sensor and attach to vehicle.
    sensor_forward_camera = world.spawn_actor(blueprint_forward, spawn_point_forward, attach_to=vehicle,
                               attachment_type=carla.AttachmentType.Rigid)
    sensor_forward_camera.listen(lambda data: sensor_callback(sensor_data=data, sensor_queue=sensor_queue, sensor_name='camera'))

    actor_list.append(sensor_forward_camera)

    # ——————————————————————————————————加载压线检测传感器——————————————————————————
    # blueprint_forward = blueprint_library.find('sensor.other.lane_invasion')
    # # Adjust sensor relative to vehicle
    # spawn_point_forward = (carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(0, 0, 0)))
    # # spawn the sensor and attach to vehicle.
    # sensor_lane_detect = world.spawn_actor(blueprint_forward, spawn_point_forward, attach_to=vehicle,
    #                                           attachment_type=carla.AttachmentType.Rigid)
    # sensor_lane_detect.listen(
    #     lambda data: sensor_callback(sensor_data=data, sensor_queue=sensor_queue, sensor_name='laneDetector'))
    #
    # actor_list.append(sensor_lane_detect)

    while (True):
        # Tick the server
        world.tick()

        spectator = world.get_spectator()
        # 将CARLA界面摄像头跟随车动
        loc = vehicle.get_transform().location
        spectator.set_transform(
            carla.Transform(carla.Location(x=loc.x, y=loc.y, z=35), carla.Rotation(yaw=0, pitch=-90, roll=0)))
        try:
            s_frame = sensor_queue.get(True, 1.0)
            print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
        except Empty:
            print("   Some of the sensor information is missed")
finally:
    # 将世界设置为异步
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
