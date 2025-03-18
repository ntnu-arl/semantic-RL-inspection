# import nvtx
import torch
import warp as wp
import math

from src.sensors.warp.warp_kernels.warp_camera_kernels import (
    DepthCameraWarpKernels,
)


class WarpCam:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        self.mesh_ids_array = mesh_ids_array

        self.width = self.cfg.width
        self.height = self.cfg.height

        self.horizontal_fov = math.radians(self.cfg.horizontal_fov_deg)
        self.far_plane = self.cfg.max_range
        self.calculate_depth = self.cfg.calculate_depth
        self.device = device

        self.camera_position_array = None
        self.camera_orientation_array = None      
        self.graph = None

        self.num_scan_lines = self.cfg.height_lidar
        self.num_points_per_line = self.cfg.width_lidar
        self.horizontal_fov_min = math.radians(self.cfg.horizontal_fov_deg_min)
        self.horizontal_fov_max = math.radians(self.cfg.horizontal_fov_deg_max)
        self.horizontal_fov_lidar = self.horizontal_fov_max - self.horizontal_fov_min
        self.horizontal_fov_mean = (self.horizontal_fov_max + self.horizontal_fov_min) / 2
        if self.horizontal_fov_lidar > 2 * math.pi:
            raise ValueError("Horizontal FOV must be less than 2pi")

        self.vertical_fov_min = math.radians(self.cfg.vertical_fov_deg_min)
        self.vertical_fov_max = math.radians(self.cfg.vertical_fov_deg_max)
        self.vertical_fov = self.vertical_fov_max - self.vertical_fov_min
        self.vertical_fov_mean = (self.vertical_fov_max + self.vertical_fov_min) / 2
        if self.vertical_fov > math.pi:
            raise ValueError("Vertical FOV must be less than pi")
        
        self.lidar_position_array = None
        self.lidar_quat_array = None
        self.graph = None
        
        self.initialize_camera_matrices()
        self.initialize_ray_vectors()

    def initialize_ray_vectors(self):
        # populate a 2D torch array with the ray vectors that are 2d arrays of wp.vec3
        ray_vectors = torch.zeros(
            (self.num_scan_lines, self.num_points_per_line, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                # Rays go from +HFoV/2 to -HFoV/2 and +VFoV/2 to -VFoV/2
                azimuth_angle = self.horizontal_fov_max - (
                    self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_points_per_line - 1))
                elevation_angle = self.vertical_fov_max - (
                    self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_scan_lines - 1))
                ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 2] = math.sin(elevation_angle)
        # normalize ray_vectors
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)

        # recast as 2D warp array of vec3
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)
        

    def initialize_camera_matrices(self):
        # Calculate camera params
        W = self.width
        H = self.height
        (u_0, v_0) = (W / 2, H / 2)
        f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)

        vertical_fov = 2 * math.atan(H / (2 * f))
        alpha_u = u_0 / math.tan(self.horizontal_fov / 2)
        alpha_v = v_0 / math.tan(vertical_fov / 2)

        # simple pinhole model
        self.K = wp.mat44(
            alpha_u,
            0.0,
            u_0,
            0.0,
            0.0,
            alpha_v,
            v_0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.K_inv = wp.inverse(self.K)

        self.c_x = int(u_0)
        self.c_y = int(v_0)

    def create_render_graph_pointcloud(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def create_render_graph_depth_range(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.face_mesh_camera == True:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range_segmentation_face_mesh,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.face_mesh_pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_occupancy_map_lidar,
                dim=(self.num_envs, self.num_scan_lines, self.num_points_per_line),
                inputs=[self.mesh_ids_array, self.lidar_position_array, self.lidar_quat_array, self.ray_vectors, self.far_plane, self.occupancy_map,
                        self.cfg.word_map_grid_size],
                device=self.device,
            )    
        elif self.cfg.segmentation_camera == True:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def set_image_tensors(self, pixels, segmentation_pixels=None, face_mesh_pixels=None, occupancy_map=None):
        # init buffers. None when uninitialized
        if self.cfg.return_pointcloud:
            self.pixels = wp.from_torch(pixels, dtype=wp.vec3)
            self.pointcloud_in_world_frame = self.cfg.pointcloud_in_world_frame
        else:
            self.pixels = wp.from_torch(pixels, dtype=wp.float32)
            
        if self.cfg.occupancy_map_lidar == True:
            self.occupancy_map = wp.from_torch(occupancy_map, dtype=wp.float32)
        else:
            self.occupancy_map = occupancy_map
            
        if self.cfg.face_mesh_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)
            self.face_mesh_pixels = wp.from_torch(face_mesh_pixels, dtype=wp.float32)
        elif self.cfg.segmentation_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)
            self.face_mesh_pixels = face_mesh_pixels
        else:
            self.segmentation_pixels = segmentation_pixels
            self.face_mesh_pixels = face_mesh_pixels

    def set_pose_tensor(self, positions, orientations, positions_lidar, orientations_lidar):
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)
        
        self.lidar_position_array = wp.from_torch(positions_lidar, dtype=wp.vec3)
        self.lidar_quat_array = wp.from_torch(orientations_lidar, dtype=wp.quat)
        
    def reset_maps(self, env_ids):
        # reset occupancy map
        occupancy_map = wp.to_torch(self.occupancy_map)
        occupancy_map[env_ids, ...] = 0.0
        self.occupancy_map = wp.from_torch(occupancy_map)
        
    # @nvtx.annotate()
    def capture(self, debug=False):
        if self.graph is None:
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud(debug=debug)
            else:
                self.create_render_graph_depth_range(debug=debug)

        if self.graph is not None:
            wp.capture_launch(self.graph)

        return wp.to_torch(self.pixels)
