# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences
import isaacgym

# isort: on

import sys

from sample_factory.enjoy import enjoy
from rl_training.train_semanticRLinspection import (
    parse_aerialgym_cfg,
    register_aerialgym_custom_components,
)

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

def main():
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
