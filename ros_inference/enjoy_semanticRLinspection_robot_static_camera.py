import sys

from ros_inference.enjoy_custom_net_static_camera_ros import NN_Inference_ROS, parse_aerialgym_cfg, quat_rotate, quat_conjugate, quat_rotate_inverse, quat_mul

import torch
# import torchvision
import numpy as np
import time

import rospy
import tf
from std_msgs.msg import UInt8MultiArray, Float32MultiArray, Header, Float64
from std_srvs.srv import Empty
from voxblox_msgs.srv import FilePath
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, PointCloud2
from mavros_msgs.msg import PositionTarget
import ros_numpy
import cv2

import numpy as np


class RlPlanner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.enable_rl = True
        
        self.use_yolo = True
        self.new_semantic = False
        # self.semantic_id_list = [-1, 1]
        self.semantic_id_list = [-1, 1, -1, 2, -1, 3]
        self.semantic_id_time_list = [45, 30, 45]
        self.semantic_index = -1
        self.semantic_start_time = 0
        self.semantic_end_time = 0
        self.in_mission = False
        self.start_mission = False
        self.changing_label = False
        rospy.set_param("/rl_policy/start_mission", False)
        rospy.set_param("/rl_policy/semantic_label", 1)
        self.semantic_id = rospy.get_param("/rl_policy/semantic_label")
        rospy.set_param("/rl_policy/next_semantic", False)
        self.next_semantic = rospy.get_param("/rl_policy/next_semantic")

        cfg = parse_aerialgym_cfg(evaluation=True)
        self.nn_model = NN_Inference_ROS(cfg)
        
        self.init_position = torch.zeros((1, 3), device=self.device, requires_grad=False)
        self.init_quats = torch.zeros((1, 4), device=self.device, requires_grad=False)

        self.image_height = 54
        self.image_width = 96

        self.local_occupancy_map = torch.zeros((21, 21, 21), device=self.device, requires_grad=False)
        self.current_height = 0.0
        self.max_height = 2.5
        # self.local_semantic_map = torch.zeros((21, 21, 21), device=self.device, requires_grad=False)
        self.agent_state = torch.zeros((1, 17), device = self.device)
        self.depth_image = -1.0 * torch.ones((self.image_height, self.image_width), device=self.device, requires_grad=False)
        self.max_depth = 10.0
        self.min_depth = 0.2
        self.segmentation_image = torch.ones((self.image_height, self.image_width), device=self.device, requires_grad=False)
        self.segmentation_image_voxblox = torch.ones((self.image_height, self.image_width), device=self.device, requires_grad=False)
        self.segmentation_depth_image_voxblox = torch.ones((self.image_height, self.image_width), device=self.device, requires_grad=False)
        self.embedding_depth_segmentation = -1.0 * torch.ones((self.image_height, self.image_width), device=self.device, requires_grad=False)
        self.n_voxels = 201
        self.sub_voxelmap_size = 21
        self.sub_voxelmap_cell_size = 0.1
        center = torch.floor(self.sub_voxelmap_size * 0.5 * torch.ones((1, 3), device = self.device))
        indices = torch.arange(self.sub_voxelmap_size, device = self.device)
        self.position_base = ((torch.cartesian_prod(indices, indices, indices)[
                              :self.sub_voxelmap_size**3, 0:3] - center) * self.sub_voxelmap_cell_size).unsqueeze(0)
        index_base = torch.cartesian_prod(indices, indices, indices)[:self.sub_voxelmap_size**3, 0:3].unsqueeze(0)
        index_base = index_base.clamp(0, self.sub_voxelmap_size - 1)
        self.sub_voxel_map_voxels_visited = torch.zeros((self.sub_voxelmap_size, self.sub_voxelmap_size, self.sub_voxelmap_size), device=self.device, requires_grad=False)
        self.idx_0_base = index_base[:, :, 0]
        self.idx_1_base = index_base[:, :, 1]
        self.idx_2_base = index_base[:, :, 2]
        
        self.entropy_position_map = torch.ones((self.n_voxels, self.n_voxels, self.n_voxels), device = self.device, requires_grad=False)
        self.depth_points = torch.zeros((self.image_width * self.image_height, 3), device = self.device, requires_grad=False)
        self.min_value = -10.0
        self.max_value = 10.0

        self.local_occupancy_map_subscriber = rospy.Subscriber(
            "/occupancy_node/local_map", UInt8MultiArray, self.localOccupancyMapCallback)
        self.odometry_subscriber = rospy.Subscriber(
            "/msf_core/odometry", Odometry, self.odometryCallback)
        
        self.action_publisher = rospy.Publisher(                        
            "/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.action_publisher_viz = rospy.Publisher(                        
            "/mavros/setpoint_raw/local_viz", TwistStamped, queue_size=1)
        self.segmentation_image_publisher = rospy.Publisher(
            "/rl_policy/segmentation_image", Image, queue_size=1) 
        # self.segmentation_image_old_publisher = rospy.Publisher(
        #     "/rl_policy/segmentation_image_old", Image, queue_size=1)
        self.rl_policy_semantic_exploration_start_publisher = rospy.Publisher(
            "/rl_policy/semantic_exploration_start", Header, queue_size=1)
        self.rl_policy_semantic_inspection_start_publisher = rospy.Publisher(
            "/rl_policy/semantic_inspection_start", Header, queue_size=1)
        
        self.mask_image_ultralytics_subscriber = rospy.Subscriber(
            "/ultralytics/mask/image", Image, self.maskImageUltralyticsCallback)
        self.mask_image_voxblox_subscriber = rospy.Subscriber(
            "/semantic_node/image_from_map", UInt8MultiArray, self.maskImageVoxbloxCallback)
        self.mask_depth_image_voxblox_subscriber = rospy.Subscriber(
            "/occupancy_node/image_from_map", UInt8MultiArray, self.maskDepthImageVoxbloxCallback)
        
        # d455 camera
        self.image_height_camera = 480
        self.image_width_camera = 640
        self.fx_camera = 386.1221923828125
        self.fy_camera = 386.1221923828125
        self.cx_camera = 318.6071472167969
        self.cy_camera = 236.15013122558594
        
        self.fx = self.fx_camera * self.image_width / self.image_width_camera
        self.fy = self.fy_camera * self.image_height / self.image_height_camera
        self.cx = self.cx_camera * self.image_width / self.image_width_camera
        self.cy = self.cy_camera * self.image_height / self.image_height_camera
        self.depth_image_topic = "/camera/depth/image_rect_raw"
        
        self.depth_image_camera = -1.0 * torch.ones((self.image_height_camera, self.image_width_camera), device=self.device, requires_grad=False)
        self.segmentation_image_camera = torch.zeros((self.image_height_camera, self.image_width_camera), device=self.device, requires_grad=False)
        # self.depth_pointcloud_ultralytics_sub = rospy.Subscriber(
        #     "/camera/stereo/points", PointCloud2, self.depthPointCloudCameraCallback)
        # self.rgb_image_camera_subscriber = rospy.Subscriber(
        #     "/camera/color/image", Image, self.rgbImageCameraCallback)
        self.depth_image_camera_subscriber = rospy.Subscriber(
            self.depth_image_topic, Image, self.depthImageCameraCallback)
        self.semantic_pointcloud_camera_publisher = rospy.Publisher(
            "/semantic_node/segmentation_pointcloud", PointCloud2, queue_size=1)
        self.depth_pointcloud_camera_publisher = rospy.Publisher(
            "/semantic_node/depth_pointcloud", Float32MultiArray, queue_size=1)
        self.depth_pointcloud_camera_check_publisher = rospy.Publisher(
            "/semantic_node/depth_pointcloud_check", PointCloud2, queue_size=1)
        self.listener = tf.TransformListener()
    
    def localOccupancyMapCallback(self, msg):
        local_occupancy_map = torch.from_numpy(np.ndarray((21, 21, 21), np.uint8, msg.data, 0)).to(self.device).float()
        # self.local_occupancy_map = torch.where(self.local_occupancy_map > 1.1, 0.0, 1.0)
        local_occupancy_map = torch.where(local_occupancy_map == 0, 0.0, local_occupancy_map)
        local_occupancy_map = torch.where(local_occupancy_map == 1, -2.0, local_occupancy_map)
        local_occupancy_map = torch.where(local_occupancy_map == 2, -1.0, local_occupancy_map)
        if self.current_height > self.max_height:
            local_occupancy_map[:, :, 16:] = -2.0
        self.local_occupancy_map = -1.0 * local_occupancy_map
    
    def odometryCallback(self, msg):
        agent_state = torch.zeros((1, 17), device = self.device)
        # position
        agent_state[..., 0] = msg.pose.pose.position.x
        agent_state[..., 1] = msg.pose.pose.position.y
        agent_state[..., 2] = msg.pose.pose.position.z

        self.current_height = msg.pose.pose.position.z

        # orientation
        agent_state[..., 3] = msg.pose.pose.orientation.x
        agent_state[..., 4] = msg.pose.pose.orientation.y
        agent_state[..., 5] = msg.pose.pose.orientation.z
        agent_state[..., 6] = msg.pose.pose.orientation.w

        # lin vel 
        agent_state[..., 7] = msg.twist.twist.linear.x
        agent_state[..., 8] = msg.twist.twist.linear.y
        agent_state[..., 9] = msg.twist.twist.linear.z

        # ang vel
        agent_state[..., 10] = msg.twist.twist.angular.x
        agent_state[..., 11] = msg.twist.twist.angular.y
        agent_state[..., 12] = msg.twist.twist.angular.z       
        
        self.agent_state = agent_state.clone()

        
    def maskImageUltralyticsCallback(self, msg):
        mask_image_ultralytics = torch.tensor(ros_numpy.numpify(msg)).to(self.device).float().unsqueeze(0).unsqueeze(0)
        self.segmentation_image_camera = torch.nn.functional.interpolate(mask_image_ultralytics, size=(self.image_height_camera, self.image_width_camera), mode='nearest-exact').squeeze()
        self.segmentation_image = torch.nn.functional.interpolate(mask_image_ultralytics, size=(self.image_height, self.image_width), mode='nearest-exact').squeeze()
        # self.segmentation_image = torchvision.transforms.functional.median_blur(segmentation_image.unsqueeze(0), kernel_size=3)
        time_yolo = msg.header.stamp.to_sec()
        time_depth = self.publish_time.to_sec()
        if abs(time_yolo - time_depth) < 0.02 and self.changing_label == False:
            points = self.generate_pointcloud_method_camera(self.depth_image_camera * self.max_depth)
            points_array = np.zeros(points.shape[0] * points.shape[1], dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('intensity', np.float32),
            ])
            semantic_mask = torch.where((self.segmentation_image_camera == 2.0) & (self.depth_image_camera < 0.25) & (self.depth_image_camera > 0.05), 1.0, 0.0)
            # semantic_mask = torch.where((self.segmentation_image_camera == 2.0) & (self.depth_image_camera < 0.55) & (self.depth_image_camera > 0.05), 1.0, 0.0)
            # print(semantic_mask.sum(dim=(0,1)))
            semantic_pointcloud = (points * semantic_mask.unsqueeze(-1)).reshape(self.image_height_camera * self.image_width_camera, 3).cpu().numpy()
            semantic_pointcloud[:, 2] = np.where((semantic_pointcloud[:, 2] < 0.5) | (semantic_pointcloud[:, 2] > 2.5), 25.0, semantic_pointcloud[:, 2])
            # semantic_pointcloud[:, 2] = np.where((semantic_pointcloud[:, 2] < 0.5) | (semantic_pointcloud[:, 2] > 5.5), 25.0, semantic_pointcloud[:, 2])
            points_array['x'] = semantic_pointcloud[:, 0]
            points_array['y'] = semantic_pointcloud[:, 1] 
            points_array['z'] = semantic_pointcloud[:, 2]
            points_array['intensity'] = 0.0
            pc_msg = ros_numpy.msgify(PointCloud2, points_array, stamp=self.publish_time, frame_id="camera_rgb_camera_optical_frame_active")
            self.semantic_pointcloud_camera_publisher.publish(pc_msg)

    def maskImageVoxbloxCallback(self, msg):
        segmentation_image_voxblox = torch.from_numpy(np.ndarray((self.image_height, self.image_width), np.uint8, msg.data, 0)).to(self.device).float()
        # print(segmentation_image_voxblox.sum(dim=))
        self.segmentation_image_voxblox = torch.where(segmentation_image_voxblox == 1.0, 2.0, 1.0)
        # segmentation_image_voxblox = torch.where(segmentation_image_voxblox == 1.0, 2.0, 1.0)
        # self.segmentation_image_voxblox = self.min_filter(segmentation_image_voxblox, kernel_size=7)

    def maskDepthImageVoxbloxCallback(self, msg):
        segmentation_depth_image_voxblox = torch.from_numpy(np.ndarray((self.image_height, self.image_width), np.uint8, msg.data, 0)).to(self.device).float()
        self.segmentation_depth_image_voxblox = torch.where(segmentation_depth_image_voxblox == 1.0, 2.0, 1.0)

    def depthImageCameraCallback(self, msg):
        # depth_image_camera = torch.from_numpy(self.fill_in_fast(ros_numpy.numpify(msg).astype('float32')/ 1000.0)).to(self.device)
        depth_image_camera = torch.from_numpy(ros_numpy.numpify(msg).astype('float32')).to(self.device) / 1000.0
        self.depth_image_camera[:] = torch.where((depth_image_camera < 0.2) | (depth_image_camera > self.max_depth), -self.max_depth, depth_image_camera)/self.max_depth
        self.depth_image = torch.nn.functional.interpolate(self.depth_image_camera.unsqueeze(0).unsqueeze(0), size=(self.image_height, self.image_width), mode='nearest-exact').squeeze()
        self.publish_time = msg.header.stamp
        
        points = self.generate_pointcloud_method(self.depth_image * self.max_depth)
        semantic_mask = torch.where((self.depth_image < 0.4) & (self.depth_image > 0.02), 1.0, 0.0)
        # semantic_mask = torch.where((self.depth_image < 0.5) & (self.depth_image > 0.02), 1.0, 0.0)
        # depth pointcloud in the camera frame
        semantic_pointcloud = (points * semantic_mask.unsqueeze(-1)).reshape(self.image_height * self.image_width, 3).cpu().numpy()
        semantic_pointcloud[:, 2] = np.where((semantic_pointcloud[:, 2] < 0.2) | (semantic_pointcloud[:, 2] > 4.0), 25.0, semantic_pointcloud[:, 2])
        # semantic_pointcloud[:, 2] = np.where((semantic_pointcloud[:, 2] < 0.2) | (semantic_pointcloud[:, 2] > 5.0), 25.0, semantic_pointcloud[:, 2])
        # depth pointcloud in the map frame        
        (trans,rot) = self.listener.lookupTransform('/map', '/camera_rgb_camera_optical_frame_active', rospy.Time(0)) # trans: x,y,z; rot: x,y,z,w
        rotated_points = self.rotate_points(semantic_pointcloud, rot) + np.array(trans)
        msg_depth_pointcloud = Float32MultiArray()
        msg_depth_pointcloud.data = np.stack((rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2])).astype('float32').ravel()
        self.depth_pointcloud_camera_publisher.publish(msg_depth_pointcloud)
        
        # points_array = np.zeros(points.shape[0] * points.shape[1], dtype=[
        #     ('x', np.float32),
        #     ('y', np.float32),
        #     ('z', np.float32),
        #     ('intensity', np.float32),
        # ])
        # points_array['x'] = rotated_points[:, 0]
        # points_array['y'] = rotated_points[:, 1] 
        # points_array['z'] = rotated_points[:, 2]
        # points_array['intensity'] = 0.0
        # pc_msg = ros_numpy.msgify(PointCloud2, points_array, stamp=msg.header.stamp, frame_id="map")
        # self.depth_pointcloud_camera_check_publisher.publish(pc_msg)

    def changeSemanticID(self, semantic_id):
        self.changing_label = True
        self.enable_rl = False
        try:
            clear_semantic_map = rospy.ServiceProxy('/semantic_node/clear_map', Empty)
            clear_semantic_map()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        rospy.set_param("/rl_policy/semantic_label", semantic_id)
        self.semantic_id = semantic_id
        self.init_position = self.agent_state[..., 0:3].clone()
        self.init_quats = self.agent_state[..., 3:7].clone()
        self.entropy_position_map[:] = 1.0
        self.nn_model.reset()
        try:
            clear_semantic_map = rospy.ServiceProxy('/semantic_node/clear_map', Empty)
            clear_semantic_map()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        if self.use_yolo == False:
            try:
                load_semantic_map = rospy.ServiceProxy('/semantic_node/load_map', FilePath)
                load_semantic_map('/home/arl/semantic_map/label'+str(self.semantic_id)+'.vxblx', )
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
        self.enable_rl = True
        rospy.loginfo("Semantic ID changed to: %s", self.semantic_id)
        self.in_mission = True
        self.new_semantic = True
        self.changing_label = False
    
    def generate_pointcloud_method_camera(self, depth_img):

        rows, cols = self.image_height_camera, self.image_width_camera
        c, r = torch.meshgrid(torch.arange(cols, device=self.device), torch.arange(rows, device=self.device), indexing='xy')
        valid = (depth_img > 0.0) & (depth_img < 10.0)
        z = torch.where(valid, depth_img, torch.nan)
        x = torch.where(valid, z * (c - self.cx_camera) / self.fx_camera, 0)
        y = torch.where(valid, z * (r - self.cy_camera) / self.fy_camera, 0)
        return torch.dstack((x, y, z))

    def generate_pointcloud_method(self, depth_img):

        rows, cols = self.image_height, self.image_width
        c, r = torch.meshgrid(torch.arange(cols, device=self.device), torch.arange(rows, device=self.device), indexing='xy')
        valid = (depth_img > 0.0) & (depth_img < 10.0)
        z = torch.where(valid, depth_img, torch.nan)
        x = torch.where(valid, z * (c - self.cx) / self.fx, 0)
        y = torch.where(valid, z * (r - self.cy) / self.fy, 0)
        return torch.dstack((x, y, z))
    
    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert a quaternion into a rotation matrix.
        """
        x, y, z, w = quat
        # Calculate elements of the rotation matrix
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rotation_matrix = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])

        return rotation_matrix
    
    def rotate_points(self, points, quat):
        """
        Rotate points by a given quaternion.

        Parameters:
            points (numpy.ndarray): The array of points with shape (5184, 3).
            quat (list or numpy.ndarray): The quaternion [x, y, z, w].

        Returns:
            numpy.ndarray: The rotated points with the same shape as the input.
        """
        rotation_matrix = self.quaternion_to_rotation_matrix(quat)
        rotated_points = np.dot(points, rotation_matrix.T)
        return rotated_points

    def median_filter_segmentation(self, image, kernel_size=3):
        padding = kernel_size // 2

        # Pad the image
        padded_image = torch.nn.functional.pad(image.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect').squeeze(0)

        # Unfold the image into sliding windows
        unfolded = torch.nn.functional.unfold(padded_image.unsqueeze(0), kernel_size=(kernel_size, kernel_size))
        unfolded = unfolded.squeeze(0).transpose(0, 1)

        # Mask for valid values (not -10.0 or -1.0)
        valid_pixels = (unfolded != -1.0)

        # Replace invalid pixels (-10.0) with NaN for median calculation
        valid_unfolded = torch.where(valid_pixels, unfolded, torch.tensor(np.nan, device=self.device))

        # Compute the median ignoring NaN values
        median_values = torch.nanmedian(valid_unfolded, dim=1).values

        # Reshape the median values back to the image shape
        median_image = median_values.view(image.shape)
        median_image = torch.nan_to_num(median_image, nan=-1.0)
        
        # Replace -1.0 pixels with their corresponding median values, but leave -10.0 unchanged
        filtered_image = torch.where(image == -1.0, median_image, image)

        return filtered_image

    def median_filter_depth(self, image, kernel_size=3):
        padding = kernel_size // 2

        # Pad the image
        padded_image = torch.nn.functional.pad(image.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect').squeeze(0)

        # Unfold the image into sliding windows
        unfolded = torch.nn.functional.unfold(padded_image.unsqueeze(0), kernel_size=(kernel_size, kernel_size))
        unfolded = unfolded.squeeze(0).transpose(0, 1)

        # Mask for valid values (not -10.0 or -1.0)
        valid_pixels = (unfolded != -1.0) & (unfolded != -10.0)

        # Replace invalid pixels (-10.0) with NaN for median calculation
        valid_unfolded = torch.where(valid_pixels, unfolded, torch.tensor(np.nan, device=self.device))

        # Compute the median ignoring NaN values
        median_values = torch.nanmedian(valid_unfolded, dim=1).values

        # Reshape the median values back to the image shape
        median_image = median_values.view(image.shape)
        median_image = torch.nan_to_num(median_image, nan=-1.0)
        
        # Replace -1.0 pixels with their corresponding median values, but leave -10.0 unchanged
        filtered_image = torch.where(image == -1.0, median_image, image)

        return filtered_image
    
    def min_filter(self, image, kernel_size=3):
        padding = kernel_size // 2

        # Pad the image
        padded_image = torch.nn.functional.pad(image.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect').squeeze(0)

        # Unfold the image into sliding windows
        unfolded = torch.nn.functional.unfold(padded_image.unsqueeze(0), kernel_size=(kernel_size, kernel_size))
        unfolded = unfolded.squeeze(0).transpose(0, 1)

        # Mask for valid values (not -10.0 or -1.0)
        valid_pixels = (unfolded != -1.0)

        # Replace invalid pixels (-10.0) with NaN for median calculation
        valid_unfolded = torch.where(valid_pixels, unfolded, torch.tensor(np.nan, device=self.device))

        # Compute the median ignoring NaN values
        min_values = torch.min(valid_unfolded, dim=1).values

        # Reshape the median values back to the image shape
        min_image = min_values.view(image.shape)
        # median_image = torch.nan_to_num(median_image, nan=-1.0)
        
        # Replace -1.0 pixels with their corresponding median values, but leave -10.0 unchanged
        # filtered_image = torch.where(image == 1.0, min_image, image)

        return min_image
    
def main():
    ema_x = 0.0
    alpha_x = 0.2
    
    ema_y = 0.0
    alpha_y = 0.2
    
    ema_z = 0.0
    alpha_z = 0.2
    
    ema_yaw = 0.0
    alpha_yaw = 0.6

    ema_camera = 0.0
    alpha_camera = 0.1

    # Initialize ROS node
    rospy.init_node('rl_planner_py')
    rl_planner = RlPlanner()
    init = True
    br = tf.TransformBroadcaster()
    
    action_mavros = PositionTarget()
    action_mavros.coordinate_frame = PositionTarget.FRAME_BODY_NED
    action_mavros.type_mask = PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ + \
                    PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + \
                    PositionTarget.IGNORE_AFZ + PositionTarget.IGNORE_YAW
    action_ros_camera = Float64()
    
    action_mavros_viz = TwistStamped()
    action_mavros_viz.header = Header(frame_id="os_sensor")
    action_camera_viz = TwistStamped()
    action_camera_viz.header = Header(frame_id="camera_rgb_camera_optical_frame_active")
    
    semantic_label_exploration_start = Header()
    semantic_label_inspection_start = Header()
    rate = rospy.Rate(10) # 10hz
    
    obs = torch.zeros((1, 17), dtype=torch.float32, requires_grad=False, device=rl_planner.device)
    obs_image = torch.zeros((1, rl_planner.image_height, rl_planner.image_width), dtype=torch.float32, requires_grad=False, device=rl_planner.device)
    obs_map = torch.zeros((1, 2, 21, 21, 21), dtype=torch.float32, requires_grad=False, device=rl_planner.device)

    rl_planner.nn_model.reset()
    obs_dict = {"obs": obs, "obs_image": obs_image, "obs_map": obs_map}
    counter = 1
    
    while not rospy.is_shutdown():
        br.sendTransform((0.0, 0.0, 0.0),
                         tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0,'ryxz'),
                         rospy.Time.now(),
                         "/camera_rgb_camera_optical_frame_active",
                         "/camera_rgb_camera_optical_frame")
        orientation_expanded = rl_planner.agent_state[..., 3:7].unsqueeze(1).expand(-1, rl_planner.sub_voxelmap_size**3, -1)
        position_expanded = rl_planner.agent_state[..., :3].unsqueeze(1).expand(-1, rl_planner.sub_voxelmap_size**3, -1)
        position_world = position_expanded + quat_rotate(orientation_expanded.reshape(-1, 4), rl_planner.position_base.reshape(-1, 3)).reshape(1, rl_planner.sub_voxelmap_size**3, 3)
        
        # segmentation_image_voxblox = rl_planner.segmentation_image_voxblox.clone()
        # segmentation_image_voxblox = torch.where((rl_planner.segmentation_image_voxblox == 2.0) & (rl_planner.segmentation_depth_image_voxblox == 2.0), 2.0, 1.0)
        embedding_depth_segmentation = torch.where((rl_planner.segmentation_image == 2.0) | (rl_planner.segmentation_image_voxblox == 2.0), 1.0, -1.0)
        # embedding_depth_segmentation = torch.where((rl_planner.segmentation_image == 2.0), 1.0, -1.0)
        embedding_depth_segmentation = rl_planner.median_filter_segmentation(embedding_depth_segmentation, kernel_size=7)
        embedding_depth_segmentation = torch.where((embedding_depth_segmentation == 1.0), rl_planner.depth_image, -10.0)
        embedding_depth_segmentation = rl_planner.median_filter_depth(embedding_depth_segmentation, kernel_size=7)
        rl_planner.embedding_depth_segmentation = torch.where(embedding_depth_segmentation == -10.0, -1.0, embedding_depth_segmentation)

        if rl_planner.enable_rl | rl_planner.start_mission:
            
            if rl_planner.new_semantic == True & (embedding_depth_segmentation > 0.0).sum(dim=(0,1)) > 260:
                rl_planner.semantic_start_time = time.time()
                rl_planner.new_semantic = False
            # update entropy map
            voxel_indices = ((rl_planner.agent_state[..., 0:3] - rl_planner.min_value) / (rl_planner.max_value - rl_planner.min_value) * (rl_planner.n_voxels - 1)).round().long()
            voxel_indices = voxel_indices.clamp(0, rl_planner.n_voxels - 1)
            rl_planner.entropy_position_map[voxel_indices[:,0],voxel_indices[:,1],voxel_indices[:,2]] += 1.0
            # get local entropy map
            voxel_indices_globalmap = ((position_world - rl_planner.min_value) / (rl_planner.max_value - rl_planner.min_value) * (rl_planner.n_voxels - 1)).round().long()
            voxel_indices_globalmap = voxel_indices_globalmap.clamp(0, rl_planner.n_voxels - 1)
            expanded_indices = voxel_indices_globalmap.view(voxel_indices_globalmap.size(0), -1, 3)
            idx_0 = expanded_indices[:, :, 0]
            idx_1 = expanded_indices[:, :, 1]
            idx_2 = expanded_indices[:, :, 2]
            values_voxels_visited = rl_planner.entropy_position_map[idx_0, idx_1, idx_2]

            probability_voxels_visited = values_voxels_visited / values_voxels_visited.sum()
            probability_voxels_visited = torch.where(probability_voxels_visited <= 0.0, 1.0, probability_voxels_visited)
            probability_voxels_visited = - probability_voxels_visited * torch.log(probability_voxels_visited)
            rl_planner.sub_voxel_map_voxels_visited[rl_planner.idx_0_base, rl_planner.idx_1_base, rl_planner.idx_2_base] = probability_voxels_visited

            # observation space -- robot state
            obs_dict["obs"] = rl_planner.agent_state.to(rl_planner.device)
            obs_dict["obs"][..., :3] = quat_rotate_inverse(rl_planner.init_quats, rl_planner.agent_state[..., 0:3] - rl_planner.init_position)
            obs_dict["obs"][...,3:7] = quat_mul(quat_conjugate(rl_planner.init_quats), rl_planner.agent_state[..., 3:7])
            # observation space -- image
            obs_dict["obs_image"] = rl_planner.embedding_depth_segmentation.unsqueeze(0)
            # observation space -- map
            obs_dict["obs_map"]   = torch.stack((rl_planner.local_occupancy_map, rl_planner.sub_voxel_map_voxels_visited), dim=0).unsqueeze(0)
            
            actions = rl_planner.nn_model.get_action(obs_dict)
            actions = np.clip(actions, -1.0, 1.0)
            rl_planner.agent_state[..., 13:17] = torch.from_numpy(actions)
            
            actions[0] *= 0.25
            actions[1] *= 0.25
            actions[2] *= 0.3
            actions[3] *= 0.3
            # actions[4] *= np.pi / 2.0
            
            # rl_planner.agent_state[..., 18] = torch.from_numpy(actions[4:])
            
            ema_x = (actions[0] * alpha_x) + (ema_x * (1 - alpha_x))
            ema_y = (actions[1] * alpha_y) + (ema_y * (1 - alpha_y))
            ema_z = (actions[2] * alpha_z) + (ema_z * (1 - alpha_z))
            ema_yaw = (actions[3] * alpha_yaw) + (ema_yaw * (1 - alpha_yaw))
            # ema_camera = (actions[4] * alpha_camera) + (ema_camera * (1 - alpha_camera))
            
            action_mavros.velocity.x = ema_x
            action_mavros.velocity.y = ema_y
            action_mavros.velocity.z = ema_z
            action_mavros.yaw_rate = ema_yaw
            
            # action_ros_camera.data = actions[4]
            
            action_mavros_viz.header.stamp = rospy.Time.now()
            action_mavros_viz.twist.linear.x = ema_x
            action_mavros_viz.twist.linear.y = ema_y
            action_mavros_viz.twist.linear.z = ema_z
            action_mavros_viz.twist.angular.z = ema_yaw
            
            # action_camera_viz.header.stamp = rospy.Time.now()
            # action_camera_viz.twist.angular.z = actions[4]
            
            counter += 1
            if init == True:
                init = False
                rl_planner.enable_rl = rospy.set_param("/rl_policy/enable_planner", False)
            if rl_planner.in_mission:
                rl_planner.semantic_end_time = time.time()
                # if rl_planner.semantic_end_time - rl_planner.semantic_start_time > rl_planner.semantic_id_time_list[rl_planner.semantic_index]:
                rl_planner.next_semantic = rospy.get_param("/rl_policy/next_semantic")
                if rl_planner.next_semantic == True:
                    rospy.set_param("/rl_policy/next_semantic", False)
                    rl_planner.next_semantic == False
                    rl_planner.semantic_index += 1
                    if rl_planner.semantic_index < len(rl_planner.semantic_id_list):
                        rl_planner.changeSemanticID(rl_planner.semantic_id_list[rl_planner.semantic_index])
                    else:
                        rl_planner.enable_rl = False
                        rospy.set_param("/rl_policy/enable_planner", False)
                        rl_planner.start_mission = False
                        rospy.set_param("/rl_policy/start_mission", False)
                        try:
                            clear_semantic_map = rospy.ServiceProxy('/semantic_node/clear_map', Empty)
                            clear_semantic_map()
                        except rospy.ServiceException as e:
                            print("Service call failed: %s"%e)
                        print("Mission END!")
        else:
            # reset RL planner if it is not in use
            rl_planner.nn_model.reset()
            rl_planner.entropy_position_map[:] = 1.0
            rl_planner.init_position = rl_planner.agent_state[..., 0:3].clone()
            rl_planner.init_quats = rl_planner.agent_state[..., 3:7].clone()
            signal_history_xyz = np.zeros((10, 3))
            signal_history_yaw = np.zeros((3, 1))
            # send zero velocity commands if not in use
            action_mavros.velocity.x = 0.0
            action_mavros.velocity.y = 0.0
            action_mavros.velocity.z = 0.0
            action_mavros.yaw_rate = 0.0

            action_ros_camera.data = 0.0
            
            action_mavros_viz.header.stamp = rospy.Time.now()
            action_mavros_viz.twist.linear.x = 0.0
            action_mavros_viz.twist.linear.y = 0.0
            action_mavros_viz.twist.linear.z = 0.0
            action_mavros_viz.twist.angular.z = 0.0
            
            action_camera_viz.header.stamp = rospy.Time.now()
            action_camera_viz.twist.angular.z = 0.0

            ema_x = 0.0
            ema_y = 0.0
            ema_z = 0.0
            ema_yaw = 0.0
            
            rl_planner.start_mission = rospy.get_param("/rl_policy/start_mission")
            if rl_planner.start_mission == True:
                rl_planner.semantic_index += 1
                rl_planner.changeSemanticID(rl_planner.semantic_id_list[rl_planner.semantic_index])
        
        rl_planner.action_publisher.publish(action_mavros)
        rl_planner.action_publisher_viz.publish(action_mavros_viz)
        rl_planner.segmentation_image_publisher.publish(ros_numpy.msgify(Image, rl_planner.embedding_depth_segmentation.cpu().numpy(), encoding="32FC1"))
        rl_planner.action_publisher_camera_viz.publish(action_camera_viz)
        
        rl_planner.enable_rl = rospy.get_param("/rl_policy/enable_planner")
        if rl_planner.start_mission == True:
            rl_planner.enable_rl = True
        if rl_planner.semantic_index < len(rl_planner.semantic_id_list):
            semantic_label_exploration_start.frame_id = str(rl_planner.semantic_id_list[rl_planner.semantic_index])
            semantic_label_exploration_start.stamp = rospy.Time.now()
            rl_planner.rl_policy_semantic_exploration_start_publisher.publish(semantic_label_exploration_start)
        semantic_label_inspection_start.frame_id = str(rl_planner.new_semantic)
        semantic_label_inspection_start.stamp = rospy.Time.now()
        rl_planner.rl_policy_semantic_inspection_start_publisher.publish(semantic_label_inspection_start)
        rospy.loginfo_throttle_identical(2, "RL-Planner is active %s", rl_planner.enable_rl)
        rate.sleep()

if __name__ == "__main__":
    sys.exit(main())
