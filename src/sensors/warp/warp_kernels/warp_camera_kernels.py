import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-1))
NO_HIT_RAY_VAL_FACE = wp.constant(-20.0)


class DepthCameraWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),
        c_x: int,
        c_y: int,
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(
            wp.transform_vector(K_inv, cam_coords_principal)
        )  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t
            mesh_obj = wp.mesh_get(mesh)
            face_index = mesh_obj.indices[f * 3]
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd
        else:
            pixels[env_id, cam_id, y, x] = dist * uv
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        c_x: int,
        c_y: int,
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(
            wp.transform_vector(K_inv, cam_coords_principal)
        )  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd
        else:
            pixels[env_id, cam_id, y, x] = dist * uv

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
        c_x: int,
        c_y: int,
        calculate_depth: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        multiplier = 1.0
        if calculate_depth:
            multiplier = wp.dot(
                rd, rd_principal
            )  # multiplier to project each ray on principal axis for depth instead of range
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t

        pixels[env_id, cam_id, y, x] = dist

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),
        c_x: int,
        c_y: int,
        calculate_depth: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        multiplier = 1.0
        if calculate_depth:
            multiplier = wp.dot(
                rd, rd_principal
            )  # multiplier to project each ray on principal axis for depth instead of range
        dist = NO_HIT_RAY_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t
            mesh_obj = wp.mesh_get(mesh)
            face_index = mesh_obj.indices[f * 3]
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])

        pixels[env_id, cam_id, y, x] = dist
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range_segmentation_face_mesh(
        mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),
        face_mesh_pixels: wp.array(dtype=float, ndim=4),
        c_x: int,
        c_y: int,
        calculate_depth: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        multiplier = 1.0
        if calculate_depth:
            multiplier = wp.dot(
                rd, rd_principal
            )  # multiplier to project each ray on principal axis for depth instead of range
        dist = NO_HIT_RAY_VAL
        segmentation_value = NO_HIT_SEGMENTATION_VAL
        face_mesh_value = NO_HIT_RAY_VAL_FACE
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t
            mesh_obj = wp.mesh_get(mesh)
            face_index = mesh_obj.indices[f * 3]
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
            face_mesh_value = float(f)

        pixels[env_id, cam_id, y, x] = dist
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value
        face_mesh_pixels[env_id, cam_id, y, x] = face_mesh_value

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_occupancy_map_lidar(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3),
        lidar_quat_array: wp.array(dtype=wp.quat),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        far_plane: float,
        occupancy_map: wp.array(dtype=float,ndim=4),
        grid_size: int,
    ):
        
        env_id, scan_line, point_index = wp.tid()
        lidar_id = env_id
        mesh = mesh_ids[env_id]
        lidar_position = lidar_pos_array[lidar_id]
        lidar_quaternion = lidar_quat_array[lidar_id]

        ray_origin = lidar_position
        ray_dir = ray_vectors[scan_line, point_index] 
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        min = wp.vec3f(-10.0, -10.0, -10.0)
        grid_size_occupancy = float(grid_size - 1)
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t

        free = float(0.025)
        distance = float(0.0)
        if dist > 3.0:
            distance = 3.0
        else:
            distance = dist
        for i in range(40):
            lidar_pointcloud_free = ray_origin + distance * free * ray_direction_world
            voxel_indices_pointcloud = ((lidar_pointcloud_free - min) / 20.0 * grid_size_occupancy)
            free += 0.025
            voxel_indices_pointcloud[0] = wp.clamp(voxel_indices_pointcloud[0], 0.0, grid_size_occupancy)
            voxel_indices_pointcloud[1] = wp.clamp(voxel_indices_pointcloud[1], 0.0, grid_size_occupancy)
            voxel_indices_pointcloud[2] = wp.clamp(voxel_indices_pointcloud[2], 0.0, grid_size_occupancy)
            occupancy_map[env_id, int(voxel_indices_pointcloud[0]), int(voxel_indices_pointcloud[1]), int(voxel_indices_pointcloud[2])] = 1.0

        # lidar_pixels[env_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
        voxel_indices_pointcloud = ((ray_origin + dist * ray_direction_world - min) / 20.0 * grid_size_occupancy)
        voxel_indices_pointcloud[0] = wp.clamp(voxel_indices_pointcloud[0], 0.0, grid_size_occupancy)
        voxel_indices_pointcloud[1] = wp.clamp(voxel_indices_pointcloud[1], 0.0, grid_size_occupancy)
        voxel_indices_pointcloud[2] = wp.clamp(voxel_indices_pointcloud[2], 0.0, grid_size_occupancy)
        occupancy_map[env_id, int(voxel_indices_pointcloud[0]), int(voxel_indices_pointcloud[1]), int(voxel_indices_pointcloud[2])] = 2.0