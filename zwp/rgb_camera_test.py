import glob
import os
import sys
import datetime

savepath = './save/'+ datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
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

IM_WIDTH = 2440
IM_HEIGHT = 1280

def process_img(image):
    global img_index
    i = np.array(image.raw_data)  # convert to an array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # was flattened, so we're going to shape it.
    i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
    cv2.imshow("", i3)  # show it.
    cv2.imwrite(os.path.join(savepath, '{}.png'.format(str(img_index))), i3)
    cv2.waitKey(1)
    img_index += 1
    return i3 / 255.0  # normalize

actor_list = []
try:
    # 获取客户
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    # 获取世界
    world = client.get_world()
    # 获取蓝图
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    # 随机选择出生点
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    #——————————————————————————————————加载前置传感器——————————————————————————
    # 加载传感器蓝图设置
    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint_forward = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint_forward.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint_forward.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint_forward.set_attribute('fov', '60')
    # Set the time in seconds between sensor captures
    blueprint_forward.set_attribute('sensor_tick', '1')
    # Adjust sensor relative to vehicle
    spawn_point_forward = (carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(0,0,0)))
    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint_forward, spawn_point_forward, attach_to=vehicle)
    def get_image_forward(data):
        savepath_for = os.path.join(savepath, 'for')
        os.makedirs(savepath_for, exist_ok=True)
        data.save_to_disk(os.path.join(savepath_for, '%d.jpg'% data.frame))
    sensor.listen(lambda data: get_image_forward(data))

    actor_list.append(sensor)


    #——————————————————————————————————加载右边传感器——————————————————————————
    # 加载传感器蓝图设置
    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint_right = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint_right.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint_right.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint_right.set_attribute('fov', '60')
    # Set the time in seconds between sensor captures
    blueprint_right.set_attribute('sensor_tick', '1')
    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(0,30,0))
    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint_right, spawn_point, attach_to=vehicle)
    def get_image_right(data):
        savepath_right = os.path.join(savepath, 'right')
        os.makedirs(savepath_right, exist_ok=True)
        data.save_to_disk(os.path.join(savepath_right, '%d.jpg'% data.frame))
    sensor.listen(lambda data: get_image_right(data))

    actor_list.append(sensor)

    #——————————————————————————————————加载右边传感器——————————————————————————
    # 加载传感器蓝图设置
    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint_left = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint_left.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint_left.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint_left.set_attribute('fov', '60')
    # Set the time in seconds between sensor captures
    blueprint_left.set_attribute('sensor_tick', '1')
    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(0,-30,0))
    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint_left, spawn_point, attach_to=vehicle)
    def get_image_left(data):
        savepath_left = os.path.join(savepath, 'left')
        os.makedirs(savepath_left, exist_ok=True)
        data.save_to_disk(os.path.join(savepath_left, '%d.jpg'% data.frame))
    sensor.listen(lambda data: get_image_right(data))

    actor_list.append(sensor)

    spectator = world.get_spectator()

    while(True):
        # Tick the server
        world.tick()

        # 将CARLA界面摄像头跟随车动
        loc = vehicle.get_transform().location
        spectator.set_transform(
            carla.Transform(carla.Location(x=loc.x, y=loc.y, z=35), carla.Rotation(yaw=0, pitch=-90, roll=0)))
finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
