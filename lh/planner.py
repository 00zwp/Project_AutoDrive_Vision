import carla
from lh.controller import VehiclePIDController
from PythonAPI.carla.agents.tools.misc import is_within_distance_ahead
from lane_detection.openvino_lane_detector import OpenVINOLaneDetector
import math
import numpy as np
import cv2
from zwp.Object_detector import Object_detector
class Planner(object):
    def __init__(self, vehicle):
        self.lane_detector = OpenVINOLaneDetector()
        self._vehicle = vehicle
        self._proximity_tlight_threshold = 5.0
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._dt = 1.0 / 20.0
        self._args_lateral_dict = {'K_P': 195, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}
        self._vehicle_controller = VehiclePIDController(vehicle=self._vehicle,
                                                          args_lateral = self._args_lateral_dict,
                                                          args_longitudinal = self._args_longitudinal_dict)

        # self.Object_detector = Object_detector()

    def run_step(self,image):

        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            control = self.emergency_stop()
        else:
            try:
                trajectory, img = self.get_trajectory_from_lane_detector(self.lane_detector, image)
                cv2.imshow("Lane", img)

                # self.Object_detector.detect(image)
                # self.Object_detector.show()
                cv2.namedWindow("traffic_light", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("traffic_light", 640, 480)
                if traffic_light is None:
                    traffic_img = cv2.imread(r"D:\Project_AutoDrive_Vision\lh\traffic_light\4.png")
                    cv2.imshow("traffic_light", traffic_img)
                else:
                    if traffic_light.state == carla.TrafficLightState.Red:
                        traffic_img = cv2.imread(r"D:\Project_AutoDrive_Vision\lh\traffic_light\1.png")
                        cv2.imshow("traffic_light", traffic_img)
                    elif traffic_light.state == carla.TrafficLightState.Green:
                        traffic_img = cv2.imread(r"D:\Project_AutoDrive_Vision\lh\traffic_light\3.png")
                        cv2.imshow("traffic_light", traffic_img)
                    elif traffic_light.state == carla.TrafficLightState.Yellow:
                        traffic_img = cv2.imread(r"D:\Project_AutoDrive_Vision\lh\traffic_light\2.png")
                        cv2.imshow("traffic_light", traffic_img)

                # 判断此处是否有车道线
                if (np.all(img[199,:] == 0)):
                    trajectory = None
            except Exception as e:
                trajectory = None
                raise e
            control = self._vehicle_controller.run_step(30, trajectory)

        return control

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1
        control.hand_brake = False
        return control


    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform,
                                        self._vehicle.get_transform(),
                                        self._proximity_tlight_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)
                else:
                    return (False, traffic_light)

        return (False, None)

    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def get_trajectory_from_lane_detector(self, lane_detector, image):
        # get lane boundaries using the lane detector
        cv2.imwrite("2.png",image)
        poly_left, poly_right, img_left, img_right = lane_detector(image)
        img = img_left + img_right
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img.astype(np.uint8)
        img = cv2.resize(img, (600, 400))

        # trajectory to follow is the mean of left and right lane boundary
        # note that we multiply with -0.5 instead of 0.5 in the formula for y below
        # according to our lane detector x is forward and y is left, but
        # according to Carla x is forward and y is right.
        x = np.arange(-2, 60, 1.0)
        y = -0.5 * (poly_left(x) + poly_right(x))
        # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
        # hence correct x coordinates
        x += 0.5
        trajectory = np.stack((x, y)).T
        return trajectory, img
