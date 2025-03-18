import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
import torch

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots.base_multirotor import BaseMultirotor


from src.task.inspection_task import InspectionTask
from src.config.task.inspection_task_config import (
    task_config as inspection_task_config,
)
from src.config.env.env_with_semantic_and_obstacles import EnvWithSemanticAndObstaclesCfg
from src.config.robot.inspection_quad_config import InspectionQuadCfg


env_config_registry.register("env_with_semantic_and_obstacles", EnvWithSemanticAndObstaclesCfg)
task_registry.register_task("inspection_task", InspectionTask, inspection_task_config)
robot_registry.register("inspection_quadrotor", BaseMultirotor, InspectionQuadCfg)


if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    rl_task_env = task_registry.make_task(
        "inspection_task",
        # other params are not set here and default values from the task config file are used
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.sim_env.robot_manager.robot.controller_config.num_actions,
        )
    ).to("cuda:0")
    actions[:] = 0.0
    with torch.no_grad():
        for i in range(10000):
            if i == 100:
                start = time.time()
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
    end = time.time()