from aerial_gym.task.base_task import BaseTask
from src.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box
import torchvision

logger = CustomLogger("inspection_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class InspectionTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device
        logger.info("Building environment for inspection task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        
        # Declare global variables for the inspection
        self.init_position = torch.zeros((self.task_config.num_envs, 3), device=self.device, requires_grad=False)
        self.init_quats = torch.zeros((self.task_config.num_envs, 4), device=self.device, requires_grad=False)
        center = torch.floor(self.task_config.sub_voxelmap_size * 0.5 * torch.ones((1, 3), device=self.device))
        # Get the indices for each dimension
        indices_0 = torch.arange(self.task_config.num_envs, device=self.device)
        indices_1 = torch.arange(self.task_config.sub_voxelmap_size, device=self.device)
        # Generate all combinations of indices using cartesian product
        index_base = torch.cartesian_prod(indices_0, indices_1, indices_1, indices_1)[:self.task_config.sub_voxelmap_size**3, 1:4].unsqueeze(0).expand(self.task_config.num_envs, -1, -1)
        self.index_base = index_base.clamp(0, self.task_config.sub_voxelmap_size - 1)
        # Get the indices for each dimension
        indices_0 = torch.arange(self.task_config.num_envs, device=self.device)
        indices_1 = torch.arange(self.task_config.sub_voxelmap_size, device=self.device)
        self.position_base = ((torch.cartesian_prod(indices_0, indices_1, indices_1, indices_1)[
                              :self.task_config.sub_voxelmap_size**3, 1:4] - center) * self.task_config.sub_voxelmap_cell_size).unsqueeze(0).expand(self.task_config.num_envs, -1, -1)
        # Init 3D maps for training
        self.min_value = self.task_config.min_value
        self.max_value = self.task_config.max_value
        self.entropy_map_position = torch.ones((self.task_config.num_envs, self.task_config.world_voxelmap_size_entropy, self.task_config.world_voxelmap_size_entropy, self.task_config.world_voxelmap_size_entropy), device=self.device, requires_grad=False)
        # Initialize the variables for face ID image 
        self.number_of_faces = 60 # needs to be the same for all SOI -- now set to 60
        self.visible_faces = torch.zeros((self.task_config.num_envs, self.number_of_faces), device=self.device, requires_grad=False)
        self.visited_faces_occurance = torch.zeros((self.task_config.num_envs, self.number_of_faces), device=self.device, requires_grad=False)
        self.face_id_mask = torch.arange(1, self.number_of_faces+1, device=self.device, requires_grad=False).unsqueeze(0).expand(self.task_config.num_envs,-1).unsqueeze(1).expand(-1, self.task_config.observation_image_dim_2d[0] * self.task_config.observation_image_dim_2d[1], -1)
        self.face_id_no_change = torch.zeros((self.task_config.num_envs), device=self.device, requires_grad=False)
        self.ema_filter = torch.zeros((self.task_config.num_envs, self.task_config.action_space_dim), device=self.device, requires_grad=False)
        self.inspection_surface_threshold = self.task_config.inspection_surface_threshold
        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self.counter = 0
        
        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {
                "obs": Box(
                    low=-np.Inf,
                    high=np.Inf,
                    shape=(self.task_config.observation_state_dim,),
                    dtype=np.float32,
                ),
                "obs_image": Box(
                    low=-np.Inf,
                    high=np.Inf,
                    shape=(1, self.task_config.observation_image_dim_2d[0], self.task_config.observation_image_dim_2d[1]),
                    dtype=np.float32,
                ),
                "obs_map": Box(
                    low=-np.Inf,
                    high=np.Inf,
                    shape=(2, self.task_config.observation_image_dim_3d[0], self.task_config.observation_image_dim_3d[1], self.task_config.observation_image_dim_3d[2]),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.task_config.action_space_dim,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "obs": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_state_dim),
                device=self.device,
                requires_grad=False,
            ),
            "obs_image": torch.zeros(
                (self.sim_env.num_envs, 1, self.task_config.observation_image_dim_2d[0], self.task_config.observation_image_dim_2d[1]),
                device=self.device,
                requires_grad=False,
            ),
            "obs_map": torch.zeros(
                (self.sim_env.num_envs, 2, self.task_config.observation_image_dim_3d[0], self.task_config.observation_image_dim_3d[1], self.task_config.observation_image_dim_3d[2]),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (self.sim_env.num_envs,  self.task_config.privileged_observation_space_dim,),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), 
                device=self.device, 
                requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), 
                device=self.device, 
                requires_grad=False
            ),
        }

        self.num_task_steps = 0

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        # Reset variables used for the inspection
        self.init_position[env_ids, ...] = self.obs_dict["robot_position"][env_ids, ...]
        self.init_quats[env_ids, ...] = self.obs_dict["robot_orientation"][env_ids, ...]
        self.entropy_map_position[env_ids, ...] = 1.0
        self.visited_faces_occurance[env_ids, ...] = 0.0
        self.face_id_no_change[env_ids] = 0.0
        self.counter = 0
        self.ema_filter[env_ids, ...] = 0.0
        self.infos = {}
        return

    def render(self):
        return self.sim_env.render()

    def logging_sanity_check(self, infos):
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occuring at the same time")
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(successes, timeouts))}"
            )
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, timeouts))}"
            )
        return

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # clamp curriculum_level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, Curriculum progress fraction: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"\nSuccess Rate: {success_rate}\nCrash Rate: {crash_rate}\nTimeout Rate: {timeout_rate}"
            )
            logger.warning(
                f"\nSuccesses: {self.success_aggregate}\nCrashes : {self.crashes_aggregate}\nTimeouts: {self.timeouts_aggregate}"
            )
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0
        
    def step(self, actions):
        self.counter += 1
        transformed_action = self.action_transformation_function(actions)
        # apply the ema filter to the actions
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)
        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # successes are are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * ( torch.where((self.face_id_no_change / self.number_of_faces > self.inspection_surface_threshold), torch.ones_like(self.truncations), torch.zeros_like(self.truncations)))
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes))
        timeouts = torch.where(self.terminations > 0, torch.zeros_like(timeouts), timeouts)  # timeouts are not counted if there is a crash

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        position_noise = (torch.rand_like(self.task_obs["obs"][...,    :3]) - 0.5) * 0.2
        orientation_noise = (torch.rand_like(self.task_obs["obs"][...,3:7]) - 0.5) * 0.2
        lin_vel_noise = (torch.rand_like(self.task_obs["obs"][..., 7:10]) - 0.5) * 0.2
        ang_vel_noise = (torch.rand_like(self.task_obs["obs"][...,10:13]) - 0.5) * 0.2
        action_noise = (torch.rand_like(self.task_obs["obs"][...,13:17]) - 0.5) * 0.2

        # Robot state observation
        self.task_obs["obs"][:, 0:3] = quat_rotate_inverse(self.init_quats, self.obs_dict["robot_position"] - self.init_position) + position_noise
        self.task_obs["obs"][:, 3:7] = quat_mul(quat_conjugate(self.init_quats), self.obs_dict["robot_orientation"]) + orientation_noise 
        self.task_obs["obs"][:, 7:10] = quat_rotate_inverse(self.obs_dict["robot_orientation"], self.obs_dict["robot_linvel"]) + lin_vel_noise 
        self.task_obs["obs"][:, 10:13] = quat_rotate_inverse(self.obs_dict["robot_orientation"], self.obs_dict["robot_angvel"]) + ang_vel_noise
        self.task_obs["obs"][:, 13:17] = self.obs_dict["robot_actions"] + action_noise
        
        # 2d image observation
        self.image_depth_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        self.image_segmentation_obs = self.obs_dict["segmentation_pixels"].squeeze(1)
        self.image_face_obs = self.obs_dict["face_mesh_pixels"].squeeze(1)
        image_semantic_depth_obs = torch.where(self.image_segmentation_obs == 100, self.image_depth_obs, -1.0)
        self.task_obs["obs_image"] = image_semantic_depth_obs.unsqueeze(1)
        
        # Update maps:
        voxel_indices = ((self.obs_dict["robot_position"] - self.task_config.min_value) / (self.task_config.max_value - self.task_config.min_value) * (self.task_config.world_voxelmap_size_entropy - 1)).round().long()
        voxel_indices = voxel_indices.clamp(0, self.task_config.world_voxelmap_size_entropy - 1)
        self.entropy_map_position[torch.arange(self.task_config.num_envs).view(-1),voxel_indices[:, 0],voxel_indices[:, 1],voxel_indices[:, 2]] += 1.0
  
        self.centered_tensors, self.centered_tensors_visited_voxels, self.centered_tensors_visit = extract_centered_tensor(
            self.obs_dict["occupancy_map"], self.entropy_map_position,
            self.task_config.sub_voxelmap_size, self.obs_dict["robot_position"], self.obs_dict["robot_orientation"], 
            self.task_config.min_value, self.task_config.max_value, self.index_base, self.position_base)
        # 3d map observation
        self.task_obs["obs_map"] = torch.stack((self.centered_tensors, self.centered_tensors_visited_voxels), dim=1)

        
    def compute_rewards_and_crashes(self, obs_dict):
        # Only consider the focus area of the camera image
        image_semantic_focus = torch.where((self.image_segmentation_obs == 100), 2.0, 1.0)
        image_semantic_focus[:, -8:, :]    = 1.0
        image_semantic_focus[:, :8, :]     = 1.0
        image_semantic_focus[:, :, -14:]   = 1.0
        image_semantic_focus[:, :, :14]    = 1.0
        camera_faces_id_entropy = torch.where((image_semantic_focus == 2.0), self.image_face_obs, -1.0).flatten(start_dim=1)
        
        # Create a mask where True indicates the matching values of each face ID
        matching_mask_entropy = camera_faces_id_entropy.unsqueeze(-1) == self.face_id_mask

        # Count the number of occurrences along dimension 1 (dimension env)       
        camera_faces_id_depth_face_mean = (matching_mask_entropy * self.image_depth_obs.flatten(start_dim=1).unsqueeze(-1)).sum(dim=1) / matching_mask_entropy.sum(dim=1)
        camera_faces_id_depth_face_mean = torch.where(torch.isnan(camera_faces_id_depth_face_mean), 10.0, camera_faces_id_depth_face_mean)
        # Per default the inspection distance is set to 1.0m by adjusting the parameter inspection_distance_shift one can adjust the inspection distance
        camera_faces_id_depth_face_mean = 1.545 * (1.25 * torch.exp(-50.0 * (camera_faces_id_depth_face_mean - self.task_config.inspection_distance_shift)**2) 
                                                 - 2.25 * torch.exp(-300.0 * (camera_faces_id_depth_face_mean-self.task_config.inspection_distance_shift)**2))
        occurrences_per_dimension_entropy = matching_mask_entropy.sum(dim=1)
        # Update the visited faces array with the new values if no collisions (or close to collisions) based on the occupancy penalty
        center = int(self.task_config.sub_voxelmap_size * 0.5)
        shift_x = self.task_config.occupancy_penalty_shift_x # shift from the (ego-centric) map center defined in number of voxels
        shift_y = self.task_config.occupancy_penalty_shift_y
        shift_z = self.task_config.occupancy_penalty_shift_z # shift in z direction from the (ego-centric) map center defined in number of voxels
        collision_mask = self.centered_tensors[..., center-shift_x:center+1+shift_x, center-shift_y:center+1+shift_y, center-shift_z:center+1+shift_z] > 1.1
        collision_mask_size = (collision_mask.shape[1] * collision_mask.shape[2] * collision_mask.shape[3])
        occupancy_penalty = collision_mask.sum(dim=(1,2,3))
        self.occupancy_penalty = torch.where(occupancy_penalty > 0.0, -(1.0 + occupancy_penalty / collision_mask_size), 0.0)
        
        occupancy_env_ids_buf = torch.where(self.occupancy_penalty < 0.0, 0.0, 1.0)
        self.occupancy_env_ids = occupancy_env_ids_buf.nonzero(as_tuple=False).squeeze(-1)
          
        self.visited_faces_occurance[self.occupancy_env_ids, ...] = torch.maximum(self.visited_faces_occurance[self.occupancy_env_ids, ...], 
                                                                             ((occurrences_per_dimension_entropy > 0) * camera_faces_id_depth_face_mean)[self.occupancy_env_ids, ...])
        # Calculate the number of faces that have not changed
        diff = self.visited_faces_occurance.sum(dim=1) - self.face_id_no_change 
        self.face_id_no_change = torch.where(diff > 0.0, self.visited_faces_occurance.sum(dim=1), self.face_id_no_change)
        face_reward = torch.where(diff > 0.0, self.visited_faces_occurance.sum(dim=1) / self.number_of_faces, 0.0)
        
        centered_tensors_visit_new = torch.where((self.centered_tensors < 1.1) & (self.centered_tensors > 0.1), self.centered_tensors_visit - 1.0, 0.0)
        centered_tensors_visit = centered_tensors_visit_new.sum(dim=(1,2,3))
        exploration_reward = torch.exp(-0.005 * ((centered_tensors_visit)**2))

        rewards = torch.where(self.occupancy_penalty < 0.0, self.occupancy_penalty, face_reward + exploration_reward * 0.01)

        return rewards, obs_dict["crashes"]

def extract_centered_tensor(global_voxel_map, global_voxel_map_entropy_visited,
                            sub_voxelmap_size, position, orientation,
                            min_value, max_value, index_base, position_base):
    
    n = global_voxel_map.shape[0]
    
    orientation_expanded = orientation.unsqueeze(1).expand(-1, sub_voxelmap_size**3, -1)
    position_expanded = position.unsqueeze(1).expand(-1, sub_voxelmap_size**3, -1)
    position_world = position_expanded + quat_rotate(orientation_expanded.reshape(-1, 4), position_base.reshape(-1, 3)).reshape(n, sub_voxelmap_size**3, 3)
        
    voxel_indices_globalmap = ((position_world - min_value) / (max_value - min_value) * (global_voxel_map.shape[1] - 1)).round().long()
    voxel_indices_globalmap = voxel_indices_globalmap.clamp(0, global_voxel_map.shape[1] - 1)
    expanded_indices = voxel_indices_globalmap.view(voxel_indices_globalmap.size(0), -1, 3)
    idx_0 = expanded_indices[:, :, 0]
    idx_1 = expanded_indices[:, :, 1]
    idx_2 = expanded_indices[:, :, 2]
    
    probability_occupancy = global_voxel_map[torch.arange(n).unsqueeze(1), idx_0, idx_1, idx_2]
    
    voxel_indices_globalmap = ((position_world - min_value) / (max_value - min_value) * (global_voxel_map_entropy_visited.shape[1] - 1)).round().long()
    voxel_indices_globalmap = voxel_indices_globalmap.clamp(0, global_voxel_map_entropy_visited.shape[1] - 1)
    expanded_indices = expanded_indices.view(voxel_indices_globalmap.size(0), -1, 3)
    idx_0 = expanded_indices[:, :, 0]
    idx_1 = expanded_indices[:, :, 1]
    idx_2 = expanded_indices[:, :, 2]
    
    values_voxels_visited = global_voxel_map_entropy_visited[torch.arange(n).unsqueeze(1), idx_0, idx_1, idx_2]
    probability_voxels_visited = values_voxels_visited / values_voxels_visited.sum(dim=(1), keepdim=True)
    probability_voxels_visited = torch.where(probability_voxels_visited <= 0.0, 1.0, probability_voxels_visited)
    probability_voxels_visited = - probability_voxels_visited * torch.log(probability_voxels_visited)
    
    sub_voxel_map = torch.zeros((n, sub_voxelmap_size, sub_voxelmap_size, sub_voxelmap_size), device=global_voxel_map.device, requires_grad=False)
    sub_voxel_map_voxels_visited = torch.zeros((n, sub_voxelmap_size, sub_voxelmap_size, sub_voxelmap_size), device=global_voxel_map.device, requires_grad=False)
    sub_voxel_map_voxels_visited_orientation = torch.zeros((n, sub_voxelmap_size, sub_voxelmap_size, sub_voxelmap_size), device=global_voxel_map.device, requires_grad=False)
    idx_0_base = index_base[:, :, 0]
    idx_1_base = index_base[:, :, 1]
    idx_2_base = index_base[:, :, 2]
    
    sub_voxel_map[torch.arange(n).unsqueeze(1), idx_0_base, idx_1_base, idx_2_base] = probability_occupancy
    sub_voxel_map_voxels_visited[torch.arange(n).unsqueeze(1), idx_0_base, idx_1_base, idx_2_base] = probability_voxels_visited
    sub_voxel_map_voxels_visited_orientation[torch.arange(n).unsqueeze(1), idx_0_base, idx_1_base, idx_2_base] = values_voxels_visited
    
    return sub_voxel_map, sub_voxel_map_voxels_visited, sub_voxel_map_voxels_visited_orientation