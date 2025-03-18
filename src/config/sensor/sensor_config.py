from aerial_gym.config.sensor_config.base_sensor_config import BaseSensorConfig
import numpy as np


class InspectionSensorConfig():
    num_sensors = 1  # number of sensors of this type

    sensor_type = "camera"  # sensor type

    height = 54
    width = 96
    horizontal_fov_deg = 87.000
    max_range = 10.0
    min_range = 0.2
    
    
    # standard OS0-128 configuration
    height_lidar = 128
    width_lidar = 512
    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180
    vertical_fov_deg_min = -45
    vertical_fov_deg_max = +45

    # Type of camera (depth, range, pointcloud, segmentation)
    # You can combine: (depth+segmentation), (range+segmentation), (pointcloud+segmentation)
    # Other combinations are trivial and you can add support for them in the code if you want.

    calculate_depth = (
        True  # Get a depth image and not a range image. False will result in a range image
    )
    return_pointcloud = False  # Return a pointcloud instead of an image. Above depth option will be ignored if this is set to True
    pointcloud_in_world_frame = False
    face_mesh_camera = True
    segmentation_camera = True
    occupancy_map_lidar = True
    word_map_grid_size = 201
    word_map_grid_size_entropy = 201
    inspection_warp = True

    # transform from sensor element coordinate frame to sensor_base_link frame
    euler_frame_rot_deg = [-90.0, 0, -90.0]
    euler_frame_rot_deg_lidar = [0.0, 0.0, 0.0]
    
    # Type of data to be returned from the sensor
    normalize_range = True  # will be set to false when pointcloud is in world frame

    # do not change this.
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # divide by max_range. Ignored when pointcloud is in world frame

    # what to do with out of range values
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0

    # randomize placement of the camera sensor
    randomize_placement = True
    # launch file position 0.27 0.07 -0.025
    # min_translation = [0.25, 0.05, -0.045] 
    # max_translation = [0.29, 0.09, -0.05]
    min_translation = [0.07, -0.05, 0.01]
    max_translation = [0.12, 0.05, 0.04]
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]
    max_euler_rotation_deg = [5.0, 5.0, 5.0]
    
    # pitching limits for the camera
    deg2rad = np.pi / 180.0
    min_pitch_rad = -np.pi / 2.0
    max_pitch_rad = np.pi / 2.0
    num_physics_steps_per_env_step_mean = 10
    time_constant = 0.5
    
    # randomize placement of the LiDAR sensor
    min_translation_lidar = [0.0, 0.0, 0.05]
    max_translation_lidar = [0.0, 0.0, 0.05]
    # example of a front-mounted dome lidar
    min_euler_rotation_deg_lidar = [0.0, 0.0, 0.0]
    max_euler_rotation_deg_lidar = [0.0, 0.0, 0.0]

    # nominal position and orientation (only for Isaac Gym Camera Sensors)
    # If you choose to use Isaac Gym sensors, their position and orientation will NOT be randomized
    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = True
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
