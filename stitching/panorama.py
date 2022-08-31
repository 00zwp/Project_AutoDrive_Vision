import glob
import os
import sys
import time
from Stitcher import Stitcher

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import numpy as np
import cv2
from queue import Queue, Empty
import random

random.seed(10)  # 决定车辆生成新位置

# args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
parser.add_argument('--port', '-p', default=2000, type=int, help='TCP port to listen to (default: 2000)')
parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager Port (default: 8000)')
parser.add_argument('--ego-spawn', type=list, default=None, help='[x,y] in world coordinate')
parser.add_argument('--top-view', default=True, help='Setting spectator to top view on ego car')
parser.add_argument('--map', default='Town04', help='Town Map')
parser.add_argument('--sync', default=True, help='Synchronous mode execution')
parser.add_argument('--sensor-h', default=2.4, help='Sensor Height')
parser.add_argument('--save-path', default='Storage', help='Synchronous mode execution')
args = parser.parse_args()

# 图片大小可自行修改
IM_WIDTH = 400
IM_HEIGHT = 800

actor_list, sensor_list = [], []
sensor_type = ['rgb']


def main(args):
    # We start creating the client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)

    world = client.get_world()
    # world = client.load_world('Town04_Opt')
    # world.unload_map_layer(carla.MapLayer.Buildings)
    # world.unload_map_layer(carla.MapLayer.Foliage )
    # world.unload_map_layer(carla.MapLayer.Ground  )
    # world.unload_map_layer(carla.MapLayer.Walls  )

    blueprint_library = world.get_blueprint_library()
    try:
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        # 手动规定
        # transform_vehicle = carla.Transform(carla.Location(0, 10, 0), carla.Rotation(0, 0, 0))
        # 自动选择
        transform_vehicle = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(random.choice(blueprint_library.filter("model3")), transform_vehicle)
        actor_list.append(ego_vehicle)

        # 设置traffic manager
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        # 是否忽略红绿灯
        # tm.ignore_lights_percentage(ego_vehicle, 100)
        # 如果限速30km/h -> 30*(1-10%)=27km/h
        tm.global_percentage_speed_difference(10.0)
        ego_vehicle.set_autopilot(True, tm.get_port())

        # -------------------------- 添加rgb相机--------------------------#
        sensor_queue = Queue()
        cam_bp = blueprint_library.find('sensor.camera.rgb')

        # 可以设置一些参数 set the attribute of camera
        cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_bp.set_attribute("fov", "55")
        # cam_bp.set_attribute('sensor_tick', '0.1')

        cam01 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=0.7, z=args.sensor_h), carla.Rotation(yaw=0)),#TODO ORIGIN X=-1.3
                                  attach_to=ego_vehicle)

        def get_image(data):
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_0/%06d.jpg' % data.frame)
            sensor_callback(data, sensor_queue, "rgb_ANGLE_0")

        cam01.listen(get_image)
        sensor_list.append(1)

        cam02 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=36)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_36")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_36/%06d.jpg' % data.frame)

        cam02.listen(get_image)
        sensor_list.append(2)

        cam03 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=72)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_72")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_72/%06d.jpg' % data.frame)

        cam03.listen(get_image)
        sensor_list.append(3)

        cam04 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=108)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_108")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_108/%06d.jpg' % data.frame)

        cam04.listen(get_image)
        sensor_list.append(4)

        cam05 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=144)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_144")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_144/%06d.jpg' % data.frame)

        cam05.listen(get_image)
        sensor_list.append(5)

        cam06 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=180)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_180")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_180/%06d.jpg' % data.frame)

        cam06.listen(get_image)
        sensor_list.append(6)

        cam07 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=216)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_216")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_216/%06d.jpg' % data.frame)

        cam07.listen(get_image)
        sensor_list.append(7)

        cam08 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=252)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_252")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_252/%06d.jpg' % data.frame)

        cam08.listen(get_image)
        sensor_list.append(8)

        cam09 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=288)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_288")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_288/%06d.jpg' % data.frame)

        cam09.listen(get_image)
        sensor_list.append(9)

        cam10 = world.spawn_actor(cam_bp,
                                  carla.Transform(carla.Location(x=1, z=args.sensor_h), carla.Rotation(yaw=324)),
                                  attach_to=ego_vehicle)

        def get_image(data):
            sensor_callback(data, sensor_queue, "rgb_ANGLE_324")
            data.save_to_disk('/home/qtk/pywork/CARLA/ANGLE_324/%06d.jpg' % data.frame)

        cam10.listen(get_image)
        sensor_list.append(10)

        # -------------------------- 设置完毕 --------------------------#

        while True:
            # Tick the server
            world.tick()

            # 将CARLA界面摄像头跟随车动
            loc = ego_vehicle.get_transform().location
            spectator.set_transform(
                carla.Transform(carla.Location(x=loc.x, y=loc.y, z=35), carla.Rotation(yaw=0, pitch=-90, roll=0)))

            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            try:
                rgbs = []
                for i in range(0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]

                    if sensor_type == 'rgb':
                        rgbs.append(_parse_image_cb(s_data))

                        # 仅用来可视化 可注释
                rgb = np.concatenate(rgbs, axis=1)[..., :3]  # 合并图像

                #cv2.imshow('vizs', visualize_data(rgb))
                #cv2.waitKey(100)
                if rgb is None or args.save_path is not None:
                    # 检查是否有各自传感器的文件夹
                    mkdir_folder(args.save_path)
                    filename = args.save_path + '/rgb/' + str(w_frame) + '.png'
                    cv2.imwrite(filename, np.array(rgb[..., ::-1]))

                Pic_name=filename[:-5]
                print("./ANGLE_0/%06d.jpg"%s_frame)
                imageA = cv2.imread("./ANGLE_0/%06d.jpg"%s_frame)
                imageB = cv2.imread("./ANGLE_36/%06d.jpg"%s_frame)
                imageC = cv2.imread("./ANGLE_72/%06d.jpg"%s_frame)
                imageD = cv2.imread("./ANGLE_108/%06d.jpg" % s_frame)
                imageE = cv2.imread("./ANGLE_144/%06d.jpg" % s_frame)
                imageF = cv2.imread("./ANGLE_180/%06d.jpg" % s_frame)
                imageG = cv2.imread("./ANGLE_216/%06d.jpg" % s_frame)
                imageH = cv2.imread("./ANGLE_252/%06d.jpg" % s_frame)
                imageI = cv2.imread("./ANGLE_288/%06d.jpg" % s_frame)
                imageJ = cv2.imread("./ANGLE_324/%06d.jpg" % s_frame)
                # 把图片拼接成全景图
                stitcher = Stitcher()
                cv2.imwrite('./imageA.jpg', imageA)
                cv2.imwrite('./imageB.jpg', imageB)
                cv2.imwrite('./imageC.jpg', imageC)
                cv2.imwrite('./imageD.jpg', imageD)
                cv2.imwrite('./imageE.jpg', imageE)
                cv2.imwrite('./imageF.jpg', imageF)
                cv2.imwrite('./imageG.jpg', imageG)
                cv2.imwrite('./imageH.jpg', imageH)
                cv2.imwrite('./imageI.jpg', imageI)
                #front 180'
                (result1, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
                (result2, vis) = stitcher.stitch([imageB, imageC], showMatches=True)
                (result3, vis) = stitcher.stitch([imageI, imageJ], showMatches=True)
                (result4, vis) = stitcher.stitch([imageJ, imageA], showMatches=True)
                (front_1, vis) = stitcher.stitch([result1, result2], showMatches=True)
                (front_2, vis) = stitcher.stitch([result4, front_1], showMatches=True)
                (front, vis) = stitcher.stitch([result3, front_2], showMatches=True)
                #behind 180'
                (result4, vis) = stitcher.stitch([imageD, imageE], showMatches=True)
                (result5, vis) = stitcher.stitch([imageE, imageF], showMatches=True)
                (result6, vis) = stitcher.stitch([imageF, imageG], showMatches=True)
                (result7, vis) = stitcher.stitch([imageG, imageH], showMatches=True)
                (behind_1, vis) = stitcher.stitch([result4, result5], showMatches=True)
                (behind_2, vis) = stitcher.stitch([result6, result7], showMatches=True)
                (behind, vis) = stitcher.stitch([behind_1, behind_2], showMatches=True)
                cv2.imwrite('./front.jpg', front)
                cv2.imwrite('./behind.jpg', behind)
                # 显示所有图片
                cv2.imshow("Keypoint Matches", vis)
                cv2.imshow("Result", front)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)
        '''for sensor in sensor_list:
            sensor.destroy()
        for actor in actor_list:
            actor.destroy()'''
        print("All cleaned up!")


def mkdir_folder(path):
    for s_type in sensor_type:
        if not os.path.isdir(os.path.join(path, s_type)):
            os.makedirs(os.path.join(path, s_type))
    return True


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))


# modify from world on rail code
def visualize_data(rgb, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)):
    canvas = np.array(rgb[..., ::-1])
    return canvas


# modify from manual control
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


if __name__ == "__main__":
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')