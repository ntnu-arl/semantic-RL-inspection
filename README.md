# semantic-RL-inspection
This repository contains the source code for the paper [_Semantically-driven Deep Reinforcement Learning for Inspection Path Planning_](https://ieeexplore.ieee.org/abstract/document/11018373). The accompanying video is available at the following [link](https://youtu.be/vAXLmalLo80?feature=shared).


[![1754819741372](https://github.com/user-attachments/assets/5988796f-017d-40a2-94db-a5b6e4224c0b)](https://youtu.be/vAXLmalLo80?feature=shared "Semantically-driven Deep Reinforcement Learning for Inspection Path Planning")

## Installation
1. Install Isaac Gym and Aerial Gym Simulator

   Follow the [instructions](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/#installation )  provided in the respective repository.
   > ### ⚠️ Important Note: Change to Argument Parser in Isaac Gym's `gymutil.py`
   >
   > Before installing the Aerial Gym Simulator, you must modify the Isaac Gym installation.
   > The argument parser in Isaac Gym may interfere with additional arguments required by other learning frameworks. To resolve this, you need to modify line 337 of the `gymutil.py` file located in the `isaacgym` folder.
   >
   > Change the following line:
   > 
   > ```python
   > args = parser.parse_args()
   > ```
   >
   > to:
   >
   > ```python
   > args, _ = parser.parse_known_args()
   > ```

2. Set up the environment

   Once the installation is successful, activate the `aerialgym` environment:
   ```bash
   cd ~/workspaces/ && conda activate aerialgym
   ```
3. Clone this repository

   Clone the repository by running the following command:
   ```bash
   git clone git@github.com:ntnu-arl/semantic-RL-inspection.git
   ```
4. Install Semantic-RL-Inspection

   Navigate to the cloned repository and install it using the following command:
   ```bash
   cd ~/workspaces/semantic-RL-inspection/
   pip install -e .
   ```
   
## Running the Examples 
The standalone examples, along with a pre-trained RL policy, can be found in the `examples` directory. The ready-to-use policy used in the work detailed in [Semantically-driven Deep Reinforcement Learning for Inspection Path Planning](https://ieeexplore.ieee.org/abstract/document/11018373) is available under `examples/pre-trained_network`. To evaluate the performance of this policy, follow the steps below.

### Single Semantic Example

This example demonstrates policy inference in a room-like environment, which was also used during training. However, in this scenario, there are no obstacles, and only a semantic object (Emerald Green) is present. To run this example, execute the following commands:
```bash
cd ~/workspaces/semantic-RL-inspection/examples/
conda activate aerialgym
bash semantic_example.sh
```
You should now be able to observe the trained policy in action — performing an inspection of the specified semantic object without any obstacles in the environment:

https://github.com/user-attachments/assets/42b8ac74-4723-4c45-857a-514ab2babb58

### Single Semantic with Obstacles Example

In this example, the policy is inferred in the same room-like environment as before, but with the addition of 4 obstacles (Tyrian Purple) alongside the semantic object (Emerald Green). To modify the number of obstacles, you can adjust the configuration in `src/config/env/env_object_config.py` by changing the value in the following class:

```python
class obstacle_asset_params(asset_state_params):
    num_assets = 4
```
To run this example, execute the following commands:
```bash
cd ~/workspaces/semantic-RL-inspection/examples/
conda activate aerialgym
bash semantic_and_obstacles_example.sh
```
You should now be able to observe the trained policy in action — inspecting the semantic object of interest (Emerald Green) while navigating around 4 obstacles (Tyrian Purple) in the environment:

https://github.com/user-attachments/assets/22320a3d-e3cf-4c11-b4e3-9958c59b702d

The default viewer is set to follow the agent. To disable this feature and inspect other parts of the environment, press `F` on your keyboard. After doing so, you will be able to observe the trained policy in action across 16 environments, each containing different semantic objects (Emerald Green) and 4 obstacles (Tyrian Purple):

https://github.com/user-attachments/assets/d29e4260-489d-4266-bfca-b7a48c96a0f2


## RL Training 
### Running Training
To train your first semantic-aware inspection policy, use the following command, which initiates the training with the settings introduced in [Semantically-driven Deep Reinforcement Learning for Inspection Path Planning](https://ieeexplore.ieee.org/abstract/document/11018373):
```bash
conda activate aerialgym
cd ~/workspaces/
python -m rl_training.train_semanticRLinspection --env=inspection_task --train_for_env_steps=100000000  --experiment=testExperiment
```
By default, the number of environments is set to 512. If your GPU cannot handle this number, reduce it by adjusting the `num_envs` parameter in `/src/config/task/inspection_task_config.py`:

```python
num_envs = 512
```

### Loading Trained Models
To load a trained checkpoint and perform only inference (no training), follow these steps:

1. For clear visualization (to avoid rendering overhead), reduce the number of environments (e.g., to 16) and enable the viewer by modifying `/src/config/task/inspection_task_config.py`:

   From:
   ```python
   num_envs = 512
   use_warp = True
   headless = True
   ```
   To:
   ```python
   num_envs = 16
   use_warp = True
   headless = False
   ```
2. For a better view during inference, consider excluding the top wall from the room-like environments by modifying the `/src/config/env/env_with_semantic_and_obstacles.py` file:

   ```python
   "top_wall": False, # excluding top wall
   ```
3. Finally, execute the inference script with the following command:

   ```bash
   conda activate aerialgym
   cd ~/workspaces/
   python -m rl_training.enjoy_semanticRLinspection --env=inspection_task --experiment=testExperiment
   ```
   The default viewer is set to follow the agent. To disable this feature and inspect other parts of the environment, press `F` on your keyboard. 

## Citing
If you reference our work in your research, please cite the following paper:

G. Malczyk, M. Kulkarni and K. Alexis, "Semantically-driven Deep Reinforcement Learning for Inspection Path Planning," Accepted for publication in IEEE Robotics and Automation Letters, 2025

```bibtex
@article{malczyk2025semantically,
  title={Semantically-Driven Deep Reinforcement Learning for Inspection Path Planning},
  author={Malczyk, Grzegorz and Kulkarni, Mihir and Alexis, Kostas},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
## Contact
For inquiries, feel free to reach out to the authors:
- **Grzegorz Malczyk**
  
  [Email](mailto:grzegorz.malczyk@ntnu.no) | [GitHub](https://github.com/grzemal) | [LinkedIn](https://www.linkedin.com/in/grzegorz-malczyk/) | [X (formerly Twitter)](https://twitter.com/grzemalige)

- **Mihir Kulkarni**

  [Email](mailto:mihirk284@gmail.com) | [GitHub](https://github.com/mihirk284) | [LinkedIn](https://www.linkedin.com/in/mihir-kulkarni-6070b6135/) | [X (formerly Twitter)](https://twitter.com/mihirk284)

- **Kostas Alexis**

  [Email](mailto:konstantinos.alexis@ntnu.no) | [GitHub](https://github.com/kostas-alexis) | [LinkedIn](https://www.linkedin.com/in/kostas-alexis-67713918/) | [X (formerly Twitter)](https://twitter.com/arlteam)

This research was conducted at the [Autonomous Robots Lab](https://www.autonomousrobotslab.com/), [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no). 

For more information, visit our website.

## Acknowledgements
This material was supported by the Research Council of Norway under Award NO-338694.

Additionally, this repository incorporates code and helper scripts from the [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator).



![arl_ntnu_logo_v2](https://github.com/user-attachments/assets/f4208309-d0a4-4084-b5aa-14adf4cb7e6c)
