from src.config.asset.env_object_config import (
    interest_asset_params,
    obstacle_asset_params,
)
from src.config.asset.env_object_config import (
    left_wall,
    right_wall,
    back_wall,
    front_wall,
    bottom_wall,
    top_wall,
)

import numpy as np


class EnvWithSemanticAndObstaclesCfg:
    class env:
        num_envs = 64  # overridden by the num_envs parameter in the task config if used
        num_env_actions = 5  # this is the number of actions handled by the environment
        # potentially some of these can be input from the RL agent for the robot and
        # some of them can be used to control various entities in the environment
        # e.g. motion of obstacles, etc.
        env_spacing = 5.0  # not used with heightfields/trimeshes

        num_physics_steps_per_env_step_mean = 10  # number of steps between camera renders mean
        num_physics_steps_per_env_step_std = 0  # number of steps between camera renders std

        render_viewer_every_n_steps = 1  # render the viewer every n steps
        reset_on_collision = (
            True  # reset environment when contact force on quadrotor is above a threshold
        )
        collision_force_threshold = 0.05  # collision force threshold [N]
        create_ground_plane = False  # create a ground plane
        sample_timestep_for_latency = True  # sample the timestep for the latency noise
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False  # write to sim at every timestep

        use_warp = True
        lower_bound_min = [-10.0, -10.0, -4.0]  # lower bound for the environment space
        lower_bound_max = [-3.0, -3.0, -1.5]  # lower bound for the environment space
        upper_bound_min = [3.0, 3.0, 1.5]  # upper bound for the environment space
        upper_bound_max = [10.0, 10.0, 4.0]  # upper bound for the environment space

    class env_config:
        include_asset_type = {
            "left_wall": True,
            "right_wall": True,
            "back_wall": True,
            "front_wall": True,
            "top_wall": True,
            "bottom_wall": True,
            "obstacle": True,
            "interest": True,
        }

        # maps the above names to the classes defining the assets. They can be enabled and disabled above in include_asset_type
        asset_type_to_dict_map = {
            "left_wall": left_wall,
            "right_wall": right_wall,
            "back_wall": back_wall,
            "front_wall": front_wall,
            "bottom_wall": bottom_wall,
            "top_wall": top_wall,
            "obstacle": obstacle_asset_params,
            "interest": interest_asset_params,
        }
