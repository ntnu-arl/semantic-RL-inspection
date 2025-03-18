import torch
from aerial_gym import AERIAL_GYM_DIRECTORY

class task_config_semantic_example:
    seed = -1
    sim_name = "base_sim"
    env_name = "env_with_semantic_example"
    robot_name = "inspection_quadrotor"
    controller_name = "lee_velocity_control"
    args = {}
    num_envs = 16
    use_warp = True
    headless = False
    device = "cuda:0"
    # state observation space
    observation_state_dim = 17
    # 2D observation space
    observation_image_dim_2d = [54, 96]
    # 3D observation space
    observation_image_dim_3d = [21, 21, 21]
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 900  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    # 3d maps params
    world_voxelmap_size = 201
    world_voxelmap_size_entropy = 201
    world_voxelmap_size_entropy_z = 201
    min_value = -10.0
    max_value = 10.0
    min_value_z = -10.0
    max_value_z = 10.0
    sub_voxelmap_size = 21
    sub_voxelmap_cell_size = 0.1
    sub_voxelmap_cell_size_entropy = 0.1
    occupancy_penalty_shift_x = 3
    occupancy_penalty_shift_y = 3
    occupancy_penalty_shift_z = 2
    inspection_surface_threshold = 0.75 # threshold for the inspection surface
    inspection_distance_shift = 0.0  # shift from the default inspection distance of 1m, the value needs to be normalized according to the max range value in sensor_config.py

    class curriculum:
        min_level = 25
        max_level = 45
        check_after_log_instances = 512
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level

    def action_transformation_function(action):
        clamped_action = torch.clamp(action, -1.0, 1.0)
        
        return clamped_action