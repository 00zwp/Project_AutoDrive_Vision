import glob
import logging
import os
import sys
import datetime
from queue import Queue, Empty

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

IM_WIDTH = 2440
IM_HEIGHT = 1280

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    if 'lidar' in sensor_name:
        sensor_data.save_to_disk(os.path.join(savepath, '%06d.ply' % sensor_data.frame))
    if 'camera' in sensor_name:
        sensor_data.save_to_disk(os.path.join(savepath, '%06d.png' % sensor_data.frame))
    if 'segmentation' in sensor_name:
        sensor_data.save_to_disk(os.path.join(savepath, '%06d.png' % sensor_data.frame))#, carla.ColorConverter.CityScapesPalette)
        # sensor_data.save_to_disk(os.path.join(savepath, '%06d-all.png' % sensor_data.frame), carla.ColorConverter.CityScapesPalette)

    sensor_queue.put((sensor_data.frame, sensor_name))


actor_list = []
try:
    # 获取客户
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    # 获取世界
    world = client.get_world()

    # 将仿真世界设置为异步状态
    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.fixed_delta_seconds = 0.00001
    # world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    sensor_queue = Queue()

    # ——————————————————————————————————加载汽车——————————————————————————
    # 获取蓝图
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    # 随机选择出生点
    # spawn_point = random.choice(world.get_map().get_spawn_points())
    spawn_point = world.get_map().get_spawn_points()[2]
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0.0, gear=1))
    actor_list.append(vehicle)

    # # ——————————————————————————————————加载语义分割器——————————————————————————
    # # 加载传感器蓝图设置
    # # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # # get the blueprint for this sensor
    # blueprint_forward = blueprint_library.find('sensor.camera.semantic_segmentation')
    # # change the dimensions of the image
    # blueprint_forward.set_attribute('image_size_x', f'{IM_WIDTH}')
    # blueprint_forward.set_attribute('image_size_y', f'{IM_HEIGHT}')
    # blueprint_forward.set_attribute('fov', '60')
    # # Set the time in seconds between sensor captures
    # blueprint_forward.set_attribute('sensor_tick', '0.5')
    # # Adjust sensor relative to vehicle
    # spawn_point_seg = carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(0, 0, 0))
    # # spawn the sensor and attach to vehicle.
    # sensor_seg = world.spawn_actor(blueprint_forward, spawn_point_seg, attach_to=vehicle,
    #                            attachment_type=carla.AttachmentType.Rigid)
    # sensor_seg.listen(lambda data: sensor_callback(sensor_data=data, sensor_queue=sensor_queue, sensor_name='segmentation'))
    # actor_list.append(sensor_seg)

    #——————————————————————————————————加载前置传感器——————————————————————————
    blueprint_forward = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint_forward.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint_forward.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint_forward.set_attribute('fov', '60')
    # Set the time in seconds between sensor captures
    blueprint_forward.set_attribute('sensor_tick', '0.001')
    # Adjust sensor relative to vehicle
    spawn_point_forward = (carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(0,0,0)))
    # spawn the sensor and attach to vehicle.
    sensor_forward_camera = world.spawn_actor(blueprint_forward, spawn_point_forward, attach_to=vehicle,
                               attachment_type=carla.AttachmentType.Rigid)
    sensor_forward_camera.listen(lambda data: sensor_callback(sensor_data=data, sensor_queue=sensor_queue, sensor_name='camera'))

    actor_list.append(sensor_forward_camera)


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
            print("Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
        except Empty:
            print("   Some of the sensor information is missed")
finally:
    # 将世界设置为异步
    # settings = world.get_settings()
    # settings.synchronous_mode = False
    # settings.fixed_delta_seconds = None
    # world.apply_settings(settings)

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
