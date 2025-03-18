import warp as wp
from src.sensors.base_sensor import BaseSensor

from aerial_gym.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    tf_apply,
    torch_rand_float_tensor,
    quat_from_euler_xyz_tensor,
    get_euler_xyz,
    get_euler_xyz_tensor,
)

import torch

from src.sensors.warp.warp_cam import WarpCam
from aerial_gym.sensors.warp.warp_lidar import WarpLidar

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("WarpSensor")


class WarpSensor(BaseSensor):
    def __init__(self, sensor_config, num_envs, mesh_id_list, device):
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.mesh_id_list = mesh_id_list
        self.device = device
        self.num_sensors = self.cfg.num_sensors

        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)

        if self.cfg.sensor_type == "lidar":
            self.sensor = WarpLidar(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Lidar sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")
        elif self.cfg.sensor_type == "camera":
            self.sensor = WarpCam(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Camera sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        else:
            raise NotImplementedError

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)
        logger.debug(f"Initializing sensor tensors")
        # for lidar
        self.robot_position_lidar = self.robot_position
        self.robot_orientation_lidar = self.robot_orientation
        # here a new view of robot position and orienentation is created since the robot has multiple sensors
        self.robot_position = self.robot_position.unsqueeze(1).expand(-1, self.num_sensors, -1)
        self.robot_orientation = self.robot_orientation.unsqueeze(1).expand(
            -1, self.num_sensors, -1
        )

        self.sensor_min_translation = torch.tensor(
            self.cfg.min_translation, device=self.device, requires_grad=False
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_max_translation = torch.tensor(
            self.cfg.max_translation, device=self.device, requires_grad=False
        ).expand(self.num_envs, self.num_sensors, -1)
                
        self.sensor_min_rotation = torch.deg2rad(
            torch.tensor(self.cfg.min_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_max_rotation = torch.deg2rad(
            torch.tensor(self.cfg.max_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, self.num_sensors, -1)
        
        euler_sensor_frame_rot = self.cfg.euler_frame_rot_deg
        sensor_frame_rot_rad = torch.deg2rad(
            torch.tensor(euler_sensor_frame_rot, device=self.device, requires_grad=False)
        )
        sensor_quat = quat_from_euler_xyz_tensor(sensor_frame_rot_rad)
        self.sensor_data_frame_quat = sensor_quat.expand(self.num_envs, self.num_sensors, -1)

        self.sensor_local_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation[..., 3] = 1.0
        self.sensor_local_orientation_euler = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation[..., 3] = 1.0
        
        # for lidar
        self.sensor_min_translation_lidar = torch.tensor(
            self.cfg.min_translation_lidar, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.sensor_max_translation_lidar = torch.tensor(
            self.cfg.max_translation_lidar, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.sensor_min_rotation_lidar = torch.deg2rad(
            torch.tensor(self.cfg.min_euler_rotation_deg_lidar, device=self.device, requires_grad=False)
        ).expand(self.num_envs, -1)
        self.sensor_max_rotation_lidar = torch.deg2rad(
            torch.tensor(self.cfg.max_euler_rotation_deg_lidar, device=self.device, requires_grad=False)
        ).expand(self.num_envs, -1)
        
        euler_sensor_frame_rot_lidar = self.cfg.euler_frame_rot_deg_lidar
        sensor_frame_rot_rad_lidar = torch.deg2rad(
            torch.tensor(euler_sensor_frame_rot_lidar, device=self.device, requires_grad=False)
        )
        sensor_quat_lidar = quat_from_euler_xyz_tensor(sensor_frame_rot_rad_lidar)
        self.sensor_data_frame_quat_lidar = sensor_quat_lidar.expand(self.num_envs, -1)
        self.sensor_local_position_lidar = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation_lidar = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation_lidar[..., 3] = 1.0
        self.sensor_position_lidar = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation_lidar = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation_lidar[..., 3] = 1.0
        
        self.sensor.set_pose_tensor(
            positions=self.sensor_position, orientations=self.sensor_orientation, positions_lidar=self.sensor_position_lidar, orientations_lidar=self.sensor_orientation_lidar
        )
        self.sensor.set_image_tensors(
            pixels=self.pixels, segmentation_pixels=self.segmentation_pixels, face_mesh_pixels=self.face_mesh_pixels, occupancy_map=self.occupancy_map
        )
        self.get_camera_local_orientation(orientation=self.camera_orientation)
        self.reset()

        logger.debug(f"[DONE] Initializing sensor tensors")

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        if self.cfg.randomize_placement == True:
            # sample local position from min and max translations
            self.sensor_local_position[env_ids] = torch_rand_float_tensor(
                self.sensor_min_translation[env_ids],
                self.sensor_max_translation[env_ids],
            )
            # sample local orientation from min and max rotations
            local_euler_rotation = torch_rand_float_tensor(
                self.sensor_min_rotation[env_ids], self.sensor_max_rotation[env_ids]
            )
            self.sensor_local_orientation[env_ids] = quat_from_euler_xyz(
                local_euler_rotation[..., 0],
                local_euler_rotation[..., 1],
                local_euler_rotation[..., 2],
            )
            self.sensor_local_orientation_euler[env_ids] = local_euler_rotation
            # for lidar
            self.sensor_local_position_lidar[env_ids] = torch_rand_float_tensor(
                self.sensor_min_translation_lidar[env_ids],
                self.sensor_max_translation_lidar[env_ids],
            )
            # sample local orientation from min and max rotations
            local_euler_rotation_lidar = torch_rand_float_tensor(
                self.sensor_min_rotation_lidar[env_ids], self.sensor_max_rotation_lidar[env_ids]
            )
            self.sensor_local_orientation_lidar[env_ids] = quat_from_euler_xyz(
                local_euler_rotation_lidar[..., 0],
                local_euler_rotation_lidar[..., 1],
                local_euler_rotation_lidar[..., 2],
            )
            self.sensor.reset_maps(env_ids)
        else:
            # Do nothing
            pass
        return

    def initialize_sensor(self):
        self.sensor.capture()

    def update(self, action):
        self.sensor_position[:] = tf_apply(
            self.robot_orientation, self.robot_position, self.sensor_local_position
        )
        
        self.sensor_orientation[:] = quat_mul(
            self.robot_orientation,
            quat_mul(self.sensor_local_orientation, self.sensor_data_frame_quat),
        )
        # for lidar
        self.sensor_position_lidar[:] = tf_apply(
            self.robot_orientation_lidar, self.robot_position_lidar, self.sensor_local_position_lidar
        )
        # quat_mul(self.root_quats, quat_mul(self.sensor_local_quat, self.correct_sensor_frame_quat))
        self.sensor_orientation_lidar[:] = quat_mul(
            self.robot_orientation_lidar,
            quat_mul(self.sensor_local_orientation_lidar, self.sensor_data_frame_quat_lidar),
        )

        logger.debug(
            f"Sensor position: {self.sensor_position[0]}, Sensor orientation: {self.sensor_orientation[0]}"
        )

        logger.debug("Capturing sensor data")
        self.sensor.capture()
        logger.debug("[DONE] Capturing sensor data")

        self.apply_noise()
        self.apply_range_limits()
        self.normalize_observation()

    def apply_range_limits(self):
        if self.cfg.return_pointcloud == True:
            # if pointcloud is in the world frame, the pointcloud range will not be normalized
            if self.cfg.pointcloud_in_world_frame == False:
                logger.debug("Pointcoud is not in world frame")
                self.pixels[
                    self.pixels.norm(dim=4, keepdim=True).expand(-1, -1, -1, -1, 3)
                    > self.cfg.max_range
                ] = self.cfg.far_out_of_range_value
                self.pixels[
                    self.pixels.norm(dim=4, keepdim=True).expand(-1, -1, -1, -1, 3)
                    < self.cfg.min_range
                ] = self.cfg.near_out_of_range_value
                logger.debug("[DONE] Clipping pointcloud values to sensor range")
        else:
            logger.debug("Pointcloud is in world frame")
            self.pixels[self.pixels > self.cfg.max_range] = self.cfg.far_out_of_range_value
            self.pixels[self.pixels < self.cfg.min_range] = self.cfg.near_out_of_range_value
            logger.debug("[DONE] Clipping pointcloud values to sensor range")

    def normalize_observation(self):
        if self.cfg.normalize_range and self.cfg.pointcloud_in_world_frame == False:
            logger.debug("Normalizing pointcloud values")
            self.pixels[:] = self.pixels / self.cfg.max_range
        if self.cfg.pointcloud_in_world_frame == True:
            logger.debug("Pointcloud is in world frame. not normalizing")

    def apply_noise(self):
        if self.cfg.sensor_noise.enable_sensor_noise == True:
            logger.debug("Applying sensor noise")
            self.pixels[:] = torch.normal(
                mean=self.pixels,
                std=self.cfg.sensor_noise.pixel_std_dev_multiplier * self.pixels,
            )
            self.pixels[
                torch.bernoulli(
                    torch.ones_like(self.pixels) * self.cfg.sensor_noise.pixel_dropout_prob
                )
                > 0
            ] = self.cfg.near_out_of_range_value

    def get_observation(self):
        return self.pixels, self.segmentation_pixels

    def get_camera_local_orientation(self, orientation):
        self.sensor_local_orientation_euler = orientation
