import sys

from ros_inference.enjoy_custom_net_static_camera_ros import NN_Inference_ROS, parse_aerialgym_cfg, quat_rotate, quat_conjugate, quat_rotate_inverse, quat_mul

import torch
import numpy as np

import rospy
from std_msgs.msg import UInt8MultiArray, Float64, Float32, Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, PointCloud2
import ros_numpy

# import torchvision
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import struct

class RlPlanner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_rl = True
        self.semantic_id = 1.0
        # register_aerialgym_custom_components()
        cfg = parse_aerialgym_cfg(evaluation=True)
        self.nn_model = NN_Inference_ROS(cfg)
        # self.observations = torch.zeros((INPUT_SIZES), device = self.device)
        # self.actions = torch.zeros((OUTPUT_SIZE), device = self.device)
        self.init_position = torch.zeros((1, 3), device=self.device, requires_grad=False)
        self.init_quats = torch.zeros((1, 4), device=self.device, requires_grad=False)

        self.image_height = 54
        self.image_width = 96

        self.local_occupancy_map = torch.zeros((21, 21, 21), device=self.device, requires_grad=False)
        self.current_height = 0.0
        self.max_height = 4.5
        self.local_semantic_map = torch.zeros((21, 21, 21), device=self.device, requires_grad=False)
        self.agent_state = torch.zeros((1, 17), device = self.device)
        self.depth_image = -1.0 * torch.ones((self.image_height, self.image_width), device=self.device, requires_grad=False)
        self.max_depth = 10.0
        self.min_depth = 0.2
        self.segmentation_image = torch.zeros((self.image_height, self.image_width), device=self.device, requires_grad=False)
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
        # self.semantic_map = torch.zeros((self.n_voxels, self.n_voxels, self.n_voxels), device = self.device, requires_grad=False)
        self.depth_points = torch.zeros((self.image_width * self.image_height, 3), device = self.device, requires_grad=False)
        self.min_value = -10.0
        self.max_value = 10.0
        # self.fx = 277
        # self.fy = 277
        # self.cx = 160
        # self.cy = 120

        self.local_occupancy_map_subscriber = rospy.Subscriber(
            "/occupancy_node/local_map", UInt8MultiArray, self.localOccupancyMapCallback)
        self.local_semantic_map_subscriber = rospy.Subscriber(
            "/semantic_node/local_map", UInt8MultiArray, self.localSemanticMapCallback)
        self.odometry_subscriber = rospy.Subscriber(
            "/rmf_owl/odometry", Odometry, self.odometryCallback)
        self.depth_image_subscriber = rospy.Subscriber(
            "/rmf_owl/front/depth_image", Image, self.depthImageCallback)
        self.segmentation_image_subscriber = rospy.Subscriber(
            "/rmf_owl/front/segmentation/labels_map", Image, self.segmentationImageCallback)
        self.semantic_id_subscriber = rospy.Subscriber(
            "semantic_id", Float32, self.semanticID)
        self.depth_pointcloud_sub = rospy.Subscriber(
            "/rmf_owl/front/depth_points", PointCloud2, self.depthPointCloudCallback)
        
        self.action_publisher = rospy.Publisher(
            "/rmf_owl/command/velocity", Twist , queue_size=1)
        self.action_publisher_viz = rospy.Publisher(                        
            "/mavros/setpoint_raw/local_viz", TwistStamped, queue_size=1)
        self.segmentation_image_publisher = rospy.Publisher(
            "/rl_policy/segmentation_image", Image, queue_size=1)
        self.semantic_pointcloud_publisher = rospy.Publisher(
            "/rl_policy/segmentation_pointcloud", PointCloud2, queue_size=1)
        self.depth_image_publisher = rospy.Publisher(
            "/rl_policy/depth_image", Image, queue_size=1)
    
    def localOccupancyMapCallback(self, msg):
        local_occupancy_map = torch.from_numpy(np.ndarray((21, 21, 21), np.uint8, msg.data, 0)).to(self.device).float()
        # self.local_occupancy_map = torch.where(self.local_occupancy_map > 1.1, 0.0, 1.0)
        local_occupancy_map = torch.where(local_occupancy_map == 0, 0.0, local_occupancy_map)
        local_occupancy_map = torch.where(local_occupancy_map == 1, -2.0, local_occupancy_map)
        local_occupancy_map = torch.where(local_occupancy_map == 2, -1.0, local_occupancy_map)
        self.local_occupancy_map = -1.0 * local_occupancy_map

    def localSemanticMapCallback(self, msg):
        local_semantic_map = torch.from_numpy(np.ndarray((21, 21, 21), np.uint8, msg.data, 0)).to(self.device).float()
        # all occupied cells belong to the semantic
        local_semantic_map = torch.where(local_semantic_map == 1, -2.0, local_semantic_map)
        # all free / unknown cells do not belong to the semantic
        local_semantic_map = torch.where(local_semantic_map == 0, -1.0, local_semantic_map)
        local_semantic_map = torch.where(local_semantic_map == 2, -1.0, local_semantic_map)
        self.local_semantic_map = -1.0 * local_semantic_map

    def odometryCallback(self, msg):
        # position
        self.agent_state[..., 0] = msg.pose.pose.position.x
        self.agent_state[..., 1] = msg.pose.pose.position.y
        self.agent_state[..., 2] = msg.pose.pose.position.z

        self.current_height = msg.pose.pose.position.z

        # orientation
        self.agent_state[..., 3] = msg.pose.pose.orientation.x
        self.agent_state[..., 4] = msg.pose.pose.orientation.y
        self.agent_state[..., 5] = msg.pose.pose.orientation.z
        self.agent_state[..., 6] = msg.pose.pose.orientation.w

        # lin vel 
        self.agent_state[..., 7] = msg.twist.twist.linear.x
        self.agent_state[..., 8] = msg.twist.twist.linear.y
        self.agent_state[..., 9] = msg.twist.twist.linear.z

        # ang vel
        self.agent_state[..., 10] = msg.twist.twist.angular.x
        self.agent_state[..., 11] = msg.twist.twist.angular.y
        self.agent_state[..., 12] = msg.twist.twist.angular.z        
    
    def depthImageCallback(self, msg):
        self.depth_image = torch.from_numpy(np.ndarray((msg.height, msg.width), np.float32, msg.data, 0)).to(self.device).float()
        self.depth_image[:] = torch.clamp(self.depth_image, 0.0, self.max_depth)/self.max_depth
        self.depth_image[self.depth_image < (self.min_depth/self.max_depth)] = -1.0
        self.depth_image_publisher.publish(ros_numpy.msgify(Image, self.depth_image.cpu().numpy(), encoding="32FC1"))

    def segmentationImageCallback(self, msg):
        segmentation_image = torch.from_numpy(np.ndarray((msg.height, msg.width, 3), np.int8, msg.data, 0))[..., 2].to(self.device).float()
        self.segmentation_image = torch.where((segmentation_image == self.semantic_id), 2.0, 1.0)

    def depthPointCloudCallback(self, msg):
        points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False)
        # points = ros_numpy.point_cloud2.pointcloud2_to_array(msg).view((np.float_, 3))
        self.depth_points = torch.nan_to_num(torch.from_numpy(points).to(self.device).float(), nan=-1.0)
        # self.depth_points[:, 0] += 0.15
        # self.depth_points[:, 2] += 0.05        
        points_array = np.zeros(points.shape[0] * points.shape[1], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
        ])
        semantic_mask = torch.where((self.segmentation_image == 2.0) & (self.depth_image < 0.12) & (self.depth_image > 0.08), 1.0, 0.0)
        semantic_pointcloud = (self.depth_points.reshape(self.image_height, self.image_width, 3) * semantic_mask.unsqueeze(-1)).reshape(self.image_height * self.image_width, 3).cpu().numpy()
        points_array['x'] = semantic_pointcloud[:, 0]
        points_array['y'] = semantic_pointcloud[:, 1]
        points_array['z'] = semantic_pointcloud[:, 2]
        points_array['intensity'] = 0.0
        pc_msg = ros_numpy.msgify(PointCloud2, points_array, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        self.semantic_pointcloud_publisher.publish(pc_msg)

    def semanticID(self, msg):
        # self.enable_rl = False
        self.semantic_id = msg.data
        self.init_position = self.agent_state[..., 0:3].clone()
        self.init_quats = self.agent_state[..., 3:7].clone()
        self.entropy_position_map[:] = 1.0
        # self.semantic_map[:] = 1.0
        self.nn_model.reset()
        # self.enable_rl = True


def main():
    signal_history_xyz = np.zeros((1, 3))
    signal_history_yaw = np.zeros((1, 1)) 
    # Initialize ROS node
    rospy.init_node('rl_planner_py')
    rl_planner = RlPlanner()
    init = True

    action_ros = Twist()
    action_ros_viz = TwistStamped()
    action_ros_viz.header = Header(frame_id="world")
    rate = rospy.Rate(10) # 10hz
    
    obs = torch.zeros((1, 17), dtype=torch.float32, requires_grad=False)
    obs_image = torch.zeros((1, rl_planner.image_height, rl_planner.image_width), dtype=torch.float32, requires_grad=False)
    obs_map = torch.zeros((1, 2, 21, 21, 21), dtype=torch.float32, requires_grad=False)

    rl_planner.nn_model.reset()
    obs_dict = {"obs": obs, "obs_image": obs_image, "obs_map": obs_map}
    counter = 1 
    
    while not rospy.is_shutdown():
        orientation_expanded = rl_planner.agent_state[..., 3:7].unsqueeze(1).expand(-1, rl_planner.sub_voxelmap_size**3, -1)
        position_expanded = rl_planner.agent_state[..., :3].unsqueeze(1).expand(-1, rl_planner.sub_voxelmap_size**3, -1)
        position_world = position_expanded + quat_rotate(orientation_expanded.reshape(-1, 4), rl_planner.position_base.reshape(-1, 3)).reshape(1, rl_planner.sub_voxelmap_size**3, 3)

        rl_planner.embedding_depth_segmentation = torch.where((rl_planner.segmentation_image == 2.0), rl_planner.depth_image, -1.0)

        if rl_planner.enable_rl:
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
            actions = np.clip(actions, -1.0, 1.0) #* 0.4
            # actions[3] *= 1.5

            rl_planner.agent_state[..., 13:17] = torch.from_numpy(actions)

            signal_history_xyz = np.roll(signal_history_xyz, -1, axis=0)
            signal_history_xyz[-1] = actions[:3]
            smoothed_signal = np.mean(signal_history_xyz, axis=0)
            
            signal_history_yaw = np.roll(signal_history_yaw, -1, axis=0)
            signal_history_yaw[-1] = actions[3]
            smoothed_signal_yaw = np.mean(signal_history_yaw, axis=0)
    
            action_ros.linear.x = smoothed_signal[0]
            action_ros.linear.y = smoothed_signal[1]
            action_ros.linear.z = smoothed_signal[2]
            action_ros.angular.z = smoothed_signal_yaw[0]
            
            action_ros_viz.header.stamp = rospy.Time.now()
            action_ros_viz.twist.linear.x = smoothed_signal[0]
            action_ros_viz.twist.linear.y = smoothed_signal[1]
            action_ros_viz.twist.linear.z = smoothed_signal[2]
            action_ros_viz.twist.angular.z = smoothed_signal_yaw[0]
            
            counter += 1
            if init == True:
                init = False
                rl_planner.enable_rl = rospy.set_param("/rl_policy/enable_planner", False)
        else:
            rl_planner.nn_model.reset()
            rl_planner.entropy_position_map[:] = 1.0
            rl_planner.init_position = rl_planner.agent_state[..., 0:3].clone()
            rl_planner.init_quats = rl_planner.agent_state[..., 3:7].clone()
            signal_history_xyz = np.zeros((1, 3))
            signal_history_yaw = np.zeros((1, 1))
            # send zero velocity commands if not in use
            action_ros.linear.x = 0.0
            action_ros.linear.y = 0.0
            action_ros.linear.z = 0.0
            action_ros.angular.z = 0.0

            action_ros_viz.header.stamp = rospy.Time.now()
            action_ros_viz.twist.linear.x = 0.0
            action_ros_viz.twist.linear.y = 0.0
            action_ros_viz.twist.linear.z = 0.0
            action_ros_viz.twist.angular.z = 0.0
            
        rl_planner.action_publisher.publish(action_ros)
        rl_planner.action_publisher_viz.publish(action_ros_viz)
        rl_planner.segmentation_image_publisher.publish(ros_numpy.msgify(Image, rl_planner.embedding_depth_segmentation.cpu().numpy(), encoding="32FC1"))

        rl_planner.enable_rl = rospy.get_param("/rl_policy/enable_planner")

        rospy.loginfo_throttle_identical(2, "RL-Planner is active %s", rl_planner.enable_rl)
        rate.sleep()

        


if __name__ == "__main__":
    sys.exit(main())
