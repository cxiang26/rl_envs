"""Gym Interface for Franka and UR"""
import os
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import gymnasium as gym
import zmq
import pickle
import cv2
import copy     
from scipy.spatial.transform import Rotation
import time
from typing import Dict


# xRocs imports (only needed for Franka/UR robots, not for A2D)
# These will be imported conditionally when needed
ConfigLoader = None
StationLoader = None
Joints = None




##############################################################################
import traceback
import sys
from rl_envs.shared_state import shared_state

def decoder_image(camera_rgb_images, camera_depth_images, bgr2rgb=False):
    if type(camera_rgb_images[0]) is np.uint8:
        rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
        if bgr2rgb and rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if camera_depth_images is not None:
            depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        else:
            depth = np.asarray([])
        return rgb, depth
    else:
        rgb_images = []
        depth_images = []
        for idx, camera_rgb_image in enumerate(camera_rgb_images):
            rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            
            if bgr2rgb and rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb)
            depth_images.append(depth)
        rgb_images = np.asarray(rgb_images)
        depth_images = np.asarray(depth_images)
        return rgb_images, depth_images
    
def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class BaseEnv(gym.Env):
    def __init__(
        self,
        fake_env=False,
        config=None,
    ):
        print('before init BaseEnv')
        self.config = config
        self._gripper_sleep = config.gripper_sleep
        self.joint_dim = config.joint_dim
        self._reset_joint = np.array(config.reset_joint)[self.joint_dim:2*self.joint_dim]
        self._reset_pose = np.array(config.reset_joint)[self.joint_dim:2*self.joint_dim]
        self._reset_left_pose = np.array(config.reset_joint)[0:self.joint_dim]
        self.use_left_arm_reset = config.use_left_arm_reset if hasattr(config, 'use_left_arm_reset') else False
        self._random_xy_range = config.random_xy_range
        self._random_rz_range = config.random_rz_range
        self._random_reset = config.random_reset
        self._bgr2rgb = config.bgr2rgb
        self._image_keys = config.image_keys
        self.robot_type = config.robot_type
        self.action_scale = config.action_scale
        self.max_episode_length = config.max_episode_length
        self.absolute_action = config.absolute_action
        self.control_mode = config.control_mode
        self.enable_rotation = config.enable_rotation
        self.interp_steps = config.interp_steps if hasattr(config, 'interp_steps') else 20  # 线性插值步数，默认10
        self.close_gripper = config.close_gripper
        self.fix_gripper = config.fix_gripper
        self.ego_mode = config.ego_mode
        self.pre_pos = None
        assert self.control_mode in ["joint", "pose"], f'Not valid control mode: {self.control_mode}'
        
        self.fake_env = fake_env
        self.hz = config.hz

        image_resize = config.image_resize
        print_green(f'in BaseEnv image_resize: {image_resize}')

        state_dict = {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "gripper_pose": gym.spaces.Box(0, 1, shape=(1,)),
                        "joints": gym.spaces.Box(
                            -np.inf, np.inf, shape=(self.joint_dim,)
                        ),
                        "ee_force": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),
                        "arm_force": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                    }

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=image_resize[key], dtype=np.uint8) 
                        for key in config.image_keys}
                ),
            }
        )

        
        if self.control_mode == "pose":
            # Action/Observation Space: action_space = (xyz + rpy + gripper)
            self.action_space = gym.spaces.Box(
                np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            )
        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.joint_dim + 1,))


        if fake_env:
            print_green("fake env : not connect to robot")
            return 

        if "a2d" in self.robot_type.lower():
            try:
                import sys
                import os
                a2d_sdk_paths = [
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'third_party'),
                    '/home/xcq/projects/HIL-RL/rl_envs',
                ]
                for path in a2d_sdk_paths:
                    if os.path.exists(path) and path not in sys.path:
                        sys.path.insert(0, path)
                
                from a2d_sdk.robot import RobotDds, CosineCamera, RobotController
                # from robot_tools.kinematics.joint2ee import Joint2EE
                # from robot_tools.kinematics.ik import InverseKinematics
                
                self.robot_station = RobotDds()
                self.robot_controller = RobotController()
                
                if hasattr(config, 'image_keys') and config.image_keys:
                    a2d_camera_names = []
                    for key in config.image_keys:
                        camera_mapping = {
                            'right': '/camera/hand_right_color',
                            'wrist': '/camera/hand_right_color',
                            'head': '/camera/head_color',
                            'left': '/camera/hand_left_color',
                        }
                        a2d_name = camera_mapping.get(key, f'/camera/{key}_color')
                        a2d_camera_names.append(a2d_name)
                    self.camera_group = CosineCamera(a2d_camera_names)
                else:
                    self.camera_group = None
                
                self.ik = None
                # if hasattr(config, 'control_mode') and config.control_mode == "pose":
                #     urdf_path = '/home/xcq/projects/HIL-RL/rl_envs/robot_tools/assets/G1_skillhands6_skillhands6_v1.5_corrected.urdf'
                    
                #     if urdf_path:
                #         self.ik = InverseKinematics(urdf_path)
                #         print_green(f"A2D: Joint2EE and IK initialized with {urdf_path}")
                #     else:
                #         print_green("A2D: Warning - URDF file not found, EE control disabled")
                
                print_green("A2D robot initialized successfully")
                
            except ImportError as e:
                print(f"[{type(e).__name__}] A2D SDK import failed: {e!r}")
                print("Please ensure a2d_sdk is installed and in PYTHONPATH")
                traceback.print_exc()
                sys.exit(1)
            except Exception as e:
                print(f"[{type(e).__name__}] A2D initialization failed: {e!r}")
                traceback.print_exc()
                sys.exit(1)
        else:
            # 原有的xRocs初始化逻辑（Franka/UR）
            try:
                from xrocs.core.config_loader import ConfigLoader
                from xrocs.core.station_loader import StationLoader
                from xrocs.common.data_type import Joints
            except ImportError as e:
                print(f"[{type(e).__name__}] xRocs import failed: {e!r}")
                print("Please ensure xRocs is installed and in PYTHONPATH")
                print("For A2D robot, xRocs is not required")
                traceback.print_exc()
                sys.exit(1)
            
            cfg_loader = ConfigLoader("/home/eai/Documents/configuration.toml")
            self.cfg_dict = cfg_loader.get_config()
            station_loader = StationLoader(self.cfg_dict)
            self.robot_station = station_loader.generate_station_handle()
            try:
                self.robot_station.connect()
            except Exception as e:
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()          # full stacktrace
                sys.exit(1)
        
        self._update_currpos()
        self.last_gripper_act = time.time()

        self.xyz_bounding_box = gym.spaces.Box(
            np.array(config.abs_pose_limit_low[:3]),
            np.array(config.abs_pose_limit_high[:3]),
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array(config.abs_pose_limit_low[3:]),
            np.array(config.abs_pose_limit_high[3:]),
            dtype=np.float64,
        )
        
        
        self.image_crop = {}

        if hasattr(config, 'image_crop') and config.image_crop is not None:
            for camera_key, crop_func in config.image_crop.items():
                if callable(crop_func):
                    self.image_crop[camera_key] = crop_func
                else:
                    def make_crop_func(crop_params):
                        crop_params = list(crop_params)
                        if isinstance(crop_params, (list, tuple)) and len(crop_params) == 2:
                            h_range, w_range = crop_params
                            return lambda img: img[h_range[0]:h_range[1], w_range[0]:w_range[1]]
                        else:
                            return lambda img: img
                    
                    self.image_crop[camera_key] = make_crop_func(crop_func)
        self.save_path = None
        self.save_frame = False
        self.obs_pre = None
        self.last_gripper_value = 1.0 if self.close_gripper else 0.0
        print('after init BaseEnv')



    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box.
        
        Args:
            pose: 位姿数组，可以是：
                - 6维: [x, y, z, roll, pitch, yaw]
                - 7维: [x, y, z, roll, pitch, yaw, gripper] 或 [x, y, z, qx, qy, qz, qw]
        
        Returns:
            裁剪后的位姿数组
        """
        original_pose = pose.copy()
        
        # 限制位置 (x, y, z)
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        
        # 限制姿态 (roll, pitch, yaw) - 仅当位姿是欧拉角格式时
        if len(pose) >= 6:
            # 判断是否是欧拉角格式（通常欧拉角范围较大，如 [-π, π]）
            # 如果是四元数格式，前3个分量通常在 [-1, 1] 范围内
            is_euler_format = (
                len(pose) == 6 or  # 6维格式通常是 xyz + rpy
                (len(pose) >= 7 and np.any(np.abs(pose[3:6]) > 1.0))  # 7维格式，如果第4-6维超出[-1,1]，可能是rpy
            )
            
            if is_euler_format:
                # 限制欧拉角
                pose[3:6] = np.clip(
                    pose[3:6], self.rpy_bounding_box.low, self.rpy_bounding_box.high
                )
        
        # 检查是否有越界（用于警告，仅在非fake_env模式下）
        if not self.fake_env and not np.array_equal(original_pose[:3], pose[:3]):
            print(f"Warning: Position clipped from {original_pose[:3]} to {pose[:3]}")
        
        return pose

    def pose_quat2euler(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_euler = Rotation.from_quat(pose_quat).as_euler("xyz")
        pose = np.hstack([pose_t, pose_euler])
        return pose
    

    def get_xtele(self) -> dict:
        if 'ur' in self.robot_type:
            self.tele_agent.switch_act()
            joints = self.tele_agent.act()
            joints = list(joints)
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv 
        elif 'franka' in self.robot_type:
            self.tele_agent.exit_any_sync()
            joints = self.tele_agent.act()
            joints = list(joints)
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv
        elif 'a2d' in self.robot_type:
            # 与 franka 保持一致：先退出同步模式，再获取动作
            # act() 返回末端位姿增量向量 [delta_pos(3) + delta_rot(3) + gripper_delta(1)] = 7维
            pose_delta = self.tele_agent.act()
            
            # 直接返回增量向量，不进行位姿计算和IK转换
            # 绝对位姿的计算应该在 HumanIntervention wrapper 中完成
            recv = {
                'joints': pose_delta.tolist(),
                'pose': pose_delta.tolist(),
            }
            return recv
        else:
            raise NotImplementedError("Unknown robot type")

    def init_xtele(self,):
        if 'ur' in self.robot_type or 'franka' in self.robot_type:
            from xtele.core.integrate_module import TeleCore
            self.tele_agent = TeleCore()
        elif 'a2d' in self.robot_type:
            # pyspacemouse 使用函数式 API，创建包装类以适配现有代码
            import pyspacemouse
            
            class SpaceMouse:
                """SpaceMouse 包装类，适配 pyspacemouse 函数式 API"""
                
                def __init__(self):
                    """初始化 SpaceMouse"""
                    self.device = pyspacemouse.open()
                    if not self.device:
                        raise RuntimeError("无法打开 SpaceMouse 设备，请确保设备已连接")
                    print("✓ SpaceMouse 已连接")
                
                def read_latest(self):
                    """
                    读取最新状态，减少缓冲区读取次数以降低延迟
                    """
                    # 快速读取3次（原来10次太多）
                    latest_state = None
                    for _ in range(3):
                        state = pyspacemouse.read()
                        if state is not None:
                            latest_state = state
                    return latest_state
                
                def get_delta(self):
                    """
                    等待并返回有效的 SpaceMouse 输入（实时，无旧缓冲数据）
                    
                    返回: dict with 'delta_pos', 'delta_rot', 'gripper_delta'
                    """
                    import time
                    
                    while True:
                        # 读取最新状态（自动丢弃旧缓冲数据）
                        state = self.read_latest()
                        if state is None:
                            raise RuntimeError("无法读取 SpaceMouse 状态")
                        
                        # 提取增量（调整坐标系：y 和 z 取反）
                        delta_pos = np.array([state.x, -state.y, -state.z])
                        delta_rot = np.array([state.roll, state.pitch, state.yaw])
                        
                        # 提取按钮状态
                        buttons = list(state.buttons)
                        if buttons[0] == 1 or buttons[1] == 1:
                            gripper_delta = 1.0
                        else:
                            gripper_delta = 0.0
                        has_button = any(b == 1 for b in buttons) if buttons else False
                        
                        # 检查是否有有效输入（降低阈值提高灵敏度）
                        has_movement = (np.abs(delta_pos).sum() > 0.1 or 
                                       np.abs(delta_rot).sum() > 0.1)
                        
                        if has_movement or has_button:
                            return {
                                'delta_pos': delta_pos,
                                'delta_rot': delta_rot,
                                'gripper_delta': gripper_delta
                            }
                        
                        time.sleep(0.001)  # 1ms 轮询间隔
                
                def act(self):
                    """返回增量向量 [dx, dy, dz, droll, dpitch, dyaw, dgripper]"""
                    delta = self.get_delta()
                    return np.concatenate([
                        delta['delta_pos'],
                        delta['delta_rot'],
                        [delta['gripper_delta']]
                    ])
                
                def sync_xtele(self, timeout=0.1):
                    """接口兼容性占位方法"""
                    pass
                
                def close(self):
                    """关闭设备"""
                    try:
                        pyspacemouse.close()
                    except:
                        pass
                
                def __del__(self):
                    """析构时关闭设备"""
                    self.close()
            
            self.tele_agent = SpaceMouse()
        else:   
            raise NotImplementedError("Unknown robot type")

    def sync_xtele(self, timeout: float = 5):
        goal = np.append(self.curr_arm_joints, self.curr_gripper_joints)

        if 'ur' in self.robot_type:
            self.tele_agent.switch_reverse()
            self.tele_agent.sync_position(goal)
        elif 'franka' in self.robot_type:
            need_torque = False
            tele_cur_joints = self.tele_agent.act()
            tele_tar_joints = goal
            timeout = int(timeout // 0.02)
            if timeout <= 1:
                path = np.array([tele_tar_joints])
            else:
                path = np.linspace(tele_cur_joints, tele_tar_joints, timeout)
            for p in path:
                self.tele_agent.sync_position_torque(p)
                time.sleep(0.02)
        elif 'a2d' in self.robot_type:
            pass
        else:
            raise NotImplementedError("Unknown robot type")    

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        
        start_time = time.time()
        if (time.time() - self.last_gripper_act > self._gripper_sleep) and not self.fix_gripper:
            include_gripper = True
        else:
            include_gripper = False

        if self.control_mode == "joint":
            obs = self._send_joint_command(action, include_gripper) 
            curr_pose_euler = None
        elif self.control_mode == "pose":
            # 首先获取当前位姿的欧拉角表示（用于插值）
            curr_pose_euler = self.pose_quat2euler(self.currpos)
            
            if self.absolute_action:
                next_pos = action
            else:
                action = action.clip(-1, 1)
                
                action_t = action[0:3] * self.action_scale[0]
                if self.enable_rotation:
                    action_euler = action[3:6] * self.action_scale[1]
                else:
                    action_euler = np.array([0, 0, 0])
                
                # Action transformation matrix
                action_mat = Rotation.from_euler("xyz", action_euler).as_matrix()
                action_pose = np.eye(4)
                action_pose[:3, :3] = action_mat
                action_pose[:3, 3] = action_t

                # 当前位置的变换矩阵
                currpos_pose = np.eye(4)
                currpos_mat = Rotation.from_quat(self.currpos[3:]).as_matrix()
                currpos_pose[:3, :3] = currpos_mat
                currpos_pose[:3, 3] = self.currpos[:3]
                
                # Calculate the new target pose (current pose × action transformation)
                tar_pose_new = currpos_pose @ action_pose
                # 计算新的目标位姿的欧拉角
                tar_euler_new = Rotation.from_matrix(tar_pose_new[:3, :3]).as_euler("xyz")

                # Calculate the new target pose (position + euler angle + gripper)
                next_pos = np.hstack([tar_pose_new[:3,3], tar_euler_new, action[-1] * self.action_scale[2]])
                next_pos = self.clip_safety_box(next_pos)
                if not self.enable_rotation:
                    if "franka" in self.robot_type:
                        next_pos[3:6] = [3.14, 0, 0]
                    else:
                        raise NotImplementedError(f"Robot {self.robot_type} does not support disable_rotation mode")
            if self.pre_pos is None:
                self.pre_pos = curr_pose_euler
            # 线性插值：从当前位置到目标位置分N步执行（N可配置，默认10）
            self._send_pos_command_with_interpolation(
                self.pre_pos, next_pos, include_gripper, interp_steps=self.interp_steps
            )
            self.pre_pos = next_pos
            # curr_pose_euler = self.pose_quat2euler(obs['arm_pose']['single'])
        else:
            raise NotImplementedError(f"Not valid control mode: {self.control_mode}")

        if include_gripper:
            self.last_gripper_act = time.time()


        self.curr_path_length += 1

        end_time = time.time()
        sleep_time = max(0.01, 1/self.hz - (end_time - start_time))
        time.sleep(sleep_time)
        obs = self._get_obs()
        curr_pose_euler = self.pose_quat2euler(obs['state']['tcp_pose'])
        
        reward = 0.0
        terminated = False
        truncated = self.curr_path_length >= self.max_episode_length
        return obs, int(reward), terminated, truncated, {"succeed": terminated, "curr_pose_euler": curr_pose_euler, "is_intervention": False}


    def reset(self, **kwargs):
        self.last_gripper_act = time.time()
        if self.ego_mode:
            # provide intervention and reset from the only one person in the scene
            self.last_gripper_value = 1.0 if self.close_gripper else 0.0
            self.sync_xtele()
            while True:
                try:
                    input("Press Enter to continue...")
                    break  # Break the loop if input is successful
                except (EOFError, ValueError):
                    print("Input is temporarily unavailable, retrying...")
                    traceback.print_exc()
                    time.sleep(1)
                    continue  
            shared_state.terminate = False
            print("Reset the scene, press Space to continue...")
            # while not shared_state.terminate:
            #     obs = self.get_xtele()
            #     xtele_joints = obs['joints']
            #     self._update_currpos()
            #     target_joint = xtele_joints.copy()
            #     self._send_joint_command(target_joint, include_gripper=True)
            #     time.sleep(1 / self.hz)
            shared_state.terminate = False
            print('go to reset!!!!!!!!!!')
            self.go_to_reset(joint_reset=True)    
        else:
            self.go_to_reset(joint_reset=True)      
            shared_state.terminate = False
            # print("重新摆放场景, 按空格继续: ")
            print("Reset the scene, press Space to continue...")
            # while not shared_state.terminate:
            #     continue
            shared_state.terminate = False

        self.pre_pos = None
        self.curr_path_length = 0
        self.last_gripper_act = time.time()
        self.last_gripper_value = 1.0 if self.close_gripper else 0.0
        time.sleep(0.5)
        obs = self._get_obs(obs=None)
        return obs, {"success": False, "is_intervention": False}


    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        # perform joint reset if needed
        self._update_currpos()
        curr_pose = self.currpos.copy()
        curr_pose = self.pose_quat2euler(curr_pose)
        reset_pose = self._reset_pose.copy()
        print("reset_pose:", reset_pose)
        # if np.linalg.norm(curr_pose - reset_pose) > 0.15 or joint_reset:
        assert self._reset_joint.shape == (self.joint_dim,)
        
        if "a2d" in self.robot_type.lower():
            try:
                # 使用末端位置控制进行重置
                use_ee_control = (
                    hasattr(self, 'robot_controller') and self.robot_controller is not None
                )

                if use_ee_control:
                    if self.use_left_arm_reset:
                        success = self._move_left_arm_to_pose_with_interpolation(
                            target_pose=self._reset_left_pose.copy(),
                            duration=3.0,
                            use_ee_control=True
                        )
                    # 使用插值函数平滑移动到重置位姿
                    success = self._move_to_pose_with_interpolation(
                        target_pose=self._reset_pose.copy(),
                        duration=3.0,
                        use_ee_control=True
                    )
                    if success:
                        print_green("A2D: Reset completed using end-effector pose control with interpolation")
                    else:
                        print(f"Warning: A2D end-effector reset with interpolation failed, falling back to joint control")
                        # use_ee_control = False
                
                # 如果末端控制失败，回退到关节控制
                if not use_ee_control:
                    # 获取当前关节状态
                    current_arm_joints, _ = self.robot_station.arm_joint_states()
                    current_waist_joints, _ = self.robot_station.waist_joint_states()
                    current_left_arm_joints = np.array(current_arm_joints[:self.joint_dim])
                    current_right_arm_joints = np.array(current_arm_joints[self.joint_dim:2*self.joint_dim])
                    
                    # 目标关节位置（右臂）
                    goal_joints = self._reset_joint.copy()
                
                    cnt = int(self.hz)
                    path = np.linspace(current_right_arm_joints, goal_joints, cnt)
                    
                    for p in path:
                        self.robot_station.move_arm(np.concatenate([current_left_arm_joints, p.tolist()]).tolist())
                        time.sleep(1 / self.hz)
                
                # 控制手部、头部和腰部
                if hasattr(self.config, 'reset_hand_positions'):
                    hand_positions = self.config.reset_hand_positions
                    # self.robot_station.move_hand(hand_positions)
                    self.robot_station.move_gripper([0.0, hand_positions[0]]) # 1.0 is the close gripper value
                    time.sleep(0.1)
                
                if hasattr(self.config, 'reset_head_positions') and hasattr(self.config, 'reset_waist_positions'):
                    self.robot_station.move_head_and_waist(
                        self.config.reset_head_positions,
                        self.config.reset_waist_positions
                    )
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Warning: A2D reset failed: {e}")
                traceback.print_exc()
        elif "ur" in self.robot_type:
            arm_joints = np.append(self.curr_arm_joints, self.last_gripper_value)
            for _ in range(5):
                self._send_joint_command(arm_joints, include_gripper=False)
                time.sleep(1 / self.hz)
                
            for name, _robot in self.robot_station.get_robot_handle().items():
                if Joints is not None:
                    goal_joints = Joints(self._reset_joint, num_of_dofs=self.joint_dim)
                else:
                    goal_joints = self._reset_joint
                try:
                    return_val =_robot.reach_target_joint(goal_joints)
                except Exception as e:
                    print(f"Error in reach_target_joint: {e}")
            for _ in range(5):
                self._send_joint_command(self._reset_joint, include_gripper=False)
                time.sleep(1 / self.hz)
        else:
            goal_joints = self._reset_joint.copy()
            # Combine the current joints and gripper position 
            curr_joints = np.concatenate([self.curr_arm_joints, np.array([self.curr_gripper_joints])])
            # Combine the target joints and gripper position
            goal_joints = np.concatenate([goal_joints, np.array([self.last_gripper_value])])
            cnt = int(3 / (1 / self.hz))
            # Generate a linear interpolation path from the current joints to the target joints (smooth transition)
            path = np.linspace(curr_joints, goal_joints, cnt)
            for p in path:
                self._send_joint_command(p, include_gripper=False)
                time.sleep(1 / self.hz)

        self._update_currpos()
        reset_pose = self.currpos.copy()
        reset_pose = self.pose_quat2euler(reset_pose)
        
        # If random reset is enabled, add random perturbations to the xy plane and rotation angle
        if self._random_reset:  
            # Add random offset to the xy plane
            reset_pose[:2] += np.random.uniform(
                -self._random_xy_range, self._random_xy_range, (2,)
            )
            # 获取旋转角
            axis_random = np.array(reset_pose[3:])
            assert axis_random.shape == (3,)
            # 在Z轴旋转角上添加随机扰动
            axis_random[-1] += np.random.uniform(
                -self._random_rz_range, self._random_rz_range
            )
            reset_pose[3:] = axis_random
            reset_pose = self.clip_safety_box(reset_pose)
            self._send_pos_command_with_interpolation(self.pose_quat2euler(self._reset_pose.copy()), reset_pose, include_gripper=False, interp_steps=50)
            # self._send_pos_command(reset_pose, include_gripper=False)


    def _send_joint_command(self, joints: np.ndarray, include_gripper=False):
        gripper_value = joints[-1] if include_gripper else self.last_gripper_value
        gripper_value_binary = 1.0 if gripper_value >= 0.5 else 0.0
        if include_gripper:
            self.last_gripper_value = gripper_value_binary
        
        if "a2d" in self.robot_type.lower():
            if len(joints) >= self.joint_dim:
                arm_joints = joints[0:self.joint_dim].tolist()
                self.robot_station.move_arm(arm_joints)
                return {
                    "arm_joints": {"single": np.array(arm_joints)},
                    "hand_joints": {"single": np.array([self.last_gripper_value])},
                    "arm_pose": {"single": self.currpos if hasattr(self, 'currpos') else np.zeros(7)},
                    "images": {}
                }
            else:
                raise ValueError(f"A2D: Expected at least {self.joint_dim} joints, got {len(joints)}")
        elif "ur" in self.robot_type:
            robot_target = {
                "arm_joints": {
                    "single": joints[0:self.joint_dim]
                },
                "hand_joints": {"single": self.last_gripper_value}
            }
            obs = self.robot_station.step(robot_target)
            return obs
        elif "franka" in self.robot_type:
            robot_target = {
                "arm": {
                    "position": {
                        "single": np.append(joints[0:self.joint_dim], self.last_gripper_value),
                    }
                }
            }
            obs = self.robot_station.step(robot_target)
            return obs
        else:
            raise NotImplementedError("Unknown robot type")


    def _send_pos_command_with_interpolation(self, curr_pose: np.ndarray, target_pose: np.ndarray, 
                                             include_gripper=False, interp_steps=10):
        """
        使用线性插值从当前位置平滑移动到目标位置
        
        Args:
            curr_pose: 当前位姿 [x, y, z, rx, ry, rz, gripper] (7维)
            target_pose: 目标位姿 [x, y, z, rx, ry, rz, gripper] (7维)
            include_gripper: 是否包含夹爪控制
            interp_steps: 插值步数，默认10步
        
        Returns:
            obs: 最后一步的观测
        """
        # 确保维度一致
        if len(curr_pose) < len(target_pose):
            # 如果当前位姿缺少夹爪维度，补0
            curr_pose = np.append(curr_pose, 0.0)
        
        # 生成平滑插值路径（使用smoothstep函数实现中间快、两端慢的过渡）
        # smoothstep函数: t^2 * (3 - 2*t)，在[0,1]范围内，t=0时速度为0，t=0.5时速度最快，t=1时速度为0
        t_linear = np.linspace(0, 1, interp_steps + 1)[1:]  # 不包含起点，只包含中间点和终点
        t_smooth = t_linear ** 2 * (3 - 2 * t_linear)  # smoothstep函数
        
        # 使用平滑后的t值进行插值
        # 只对前6个维度（位置和旋转）进行插值，最后一个维度（gripper）保持target_pose的值不变
        interp_path = []
        for t in t_smooth:
            interpolated_pose = curr_pose.copy()
            # 对前6个维度进行插值
            interpolated_pose[:6] = curr_pose[:6] + (target_pose[:6] - curr_pose[:6]) * t
            # 最后一个维度（gripper）保持target_pose的值不变
            if len(target_pose) > 6:
                interpolated_pose[6] = target_pose[6]
            interp_path.append(interpolated_pose)
        interp_path = np.array(interp_path)
        
        obs = None
        for i, interpolated_pose in enumerate(interp_path):
            # 对每个插值点发送命令
            self._send_pos_command(interpolated_pose, include_gripper)
            
            # 最后一步不需要等待
            if i < len(interp_path) - 1:
                time.sleep(0.01)  # 1ms延迟，保证机器人响应
    
    def _send_pos_command(self, pose: np.ndarray, include_gripper=False):
        if include_gripper:
            gripper_value_binary = 1.0 if pose[-1] >= 0.5 else 0.0
            self.last_gripper_value = gripper_value_binary

        if "a2d" in self.robot_type.lower():
            if len(pose) < 6:
                print("Warning: A2D pose control requires at least 6 dimensions (xyz + rpy)")
                return {
                    "arm_joints": {"single": np.zeros(self.joint_dim)},
                    "hand_joints": {"single": np.array([self.last_gripper_value])},
                    "arm_pose": {"single": np.append(pose[:3] if len(pose) >= 3 else [0, 0, 0], [0, 0, 0, 1])},
                    "images": {}
                }
            
            # 在发送给机器人之前，先进行边界限制
            # 注意：这里 pose 可能是 6维 (xyz + rpy) 或 7维 (xyz + rpy + gripper)
            pose_clipped = self.clip_safety_box(pose.copy())
            
            # 转换位姿格式：6维 (xyz + rpy) 或 7维 (xyz + quat) -> 7维 (xyz + quat)
            if len(pose_clipped) == 6:
                pos = pose_clipped[:3]
                euler = pose_clipped[3:6]
                quat = Rotation.from_euler("xyz", euler).as_quat()
                target_pose = np.concatenate([pos, quat])
            else:
                # 7维格式：可能是 xyz + rpy + gripper 或 xyz + quat + gripper
                # 检查第4-6维是否是欧拉角（通常范围较大）还是四元数（通常在[-1,1]）
                if np.all(np.abs(pose_clipped[3:6]) <= 1.0) and len(pose_clipped) == 7:
                    # 可能是四元数格式（前3个分量）
                    target_pose = pose_clipped[:7]
                else:
                    # 可能是欧拉角格式，需要转换
                    pos = pose_clipped[:3]
                    euler = pose_clipped[3:6]
                    quat = Rotation.from_euler("xyz", euler).as_quat()
                    target_pose = np.concatenate([pos, quat])
            
            # 构建位姿字典
            right_pose = {
                'x': target_pose[0], 'y': target_pose[1], 'z': target_pose[2],
                'qx': target_pose[3], 'qy': target_pose[4], 'qz': target_pose[5], 'qw': target_pose[6]
            }
            
            # 优先使用 robot_controller 进行末端位姿控制
            use_robot_controller = hasattr(self, 'robot_controller') and self.robot_controller is not None
            
            if use_robot_controller:
                try:
                    # 计算 lifetime（基于控制频率，通常设置为 1-2 个控制周期）
                    lifetime = max(1.0 / self.hz, 0.05)  # 至少 0.1 秒
                    
                    # 使用 robot_controller 进行末端位姿控制
                    self.robot_controller.set_end_effector_pose_control(
                        lifetime=lifetime,
                        control_group=['right_arm'],
                        left_pose=None,
                        right_pose=right_pose
                    )
                    self.robot_station.move_gripper([0.0, self.last_gripper_value])
                except Exception as e:
                    print(f"Warning: A2D robot_controller pose control failed: {e}, falling back to IK")
                    # use_robot_controller = False
            
            # 如果 robot_controller 不可用或失败，回退到手动 IK 方案
            if not use_robot_controller:
                if hasattr(self, 'ik') and self.ik is not None:
                    try:
                        arm_joints, _ = self.robot_station.arm_joint_states()
                        waist_joints, _ = self.robot_station.waist_joint_states()
                        current_joints = list(arm_joints) + list(waist_joints)
                        
                        joint_command = self.ik.compute_inverse_kinematics(
                            control_group=['right_arm'],
                            left_pose=None,
                            right_pose=right_pose,
                            initial_joints=current_joints
                        )
                        
                        if joint_command is not None:
                            self.robot_station.move_arm(joint_command[:14].tolist())
                        else:
                            print("Warning: A2D IK solve failed, skipping action")
                    except Exception as e:
                        print(f"Warning: A2D IK-based pose control failed: {e}")
                else:
                    print("Warning: A2D end-effector control requires robot_controller or IK solver, currently disabled")
            obs = self._get_obs()
            return {
                "arm_joints": {"single": obs['state']['joints']},
                "hand_joints": {"single": np.array([self.last_gripper_value])},
                "arm_pose": {"single": obs['state']['tcp_pose']},
                "images": obs['images']
            }
        elif "ur" in self.robot_type:
            robot_target = {
                "arm_pose": {
                    "single": pose[0:6]
                },
                "hand_joints": {"single": self.last_gripper_value}
            }
            obs = self.robot_station.step_ee(robot_target)
        else:
            arm_pose = np.append(pose[0:6], self.last_gripper_value)
            robot_target = {
                "arm_pose": {
                    "single": arm_pose
                },
                "hand_joints": {}
            }
            if "franka" in self.robot_type:
                currpose = self.currpos.copy()
                currpose_t, currpose_quat = currpose[0:3], currpose[3:]
                currpose_quat = Rotation.from_quat(currpose_quat).as_euler("xyz")
                currpose = np.hstack([currpose_t, currpose_quat])
                obs = self.robot_station.step_ee(robot_target)
            else:
                raise NotImplementedError("Unknown robot type")
        return obs


    def _get_current_ee_pose(self, arm: str = 'right') -> np.ndarray:
        """
        获取当前末端位姿的统一方法。

        Args:
            arm: 'left' 或 'right'，默认 'right'。

        Returns:
            np.ndarray: 当前末端位姿 [x, y, z, qx, qy, qz, qw]
        """
        frame_key = 'arm_left_link7' if arm == 'left' else 'arm_right_link7'
        current_states = self.robot_controller.get_motion_status()
        ee_frame = current_states['frames'][frame_key]
        pos, quat = ee_frame['position'], ee_frame['orientation']['quaternion']
        return np.array([
            pos['x'], pos['y'], pos['z'],
            quat['x'], quat['y'], quat['z'], quat['w']
        ])


    def _move_to_pose_with_interpolation(
        self, 
        target_pose: np.ndarray, 
        duration: float = 3.0,
        use_ee_control: bool = True
    ) -> bool:
        """
        使用插值平滑移动到目标位姿
        
        Args:
            target_pose: 目标位姿，可以是：
                - 7维: [x, y, z, qx, qy, qz, qw] (四元数格式)
                - 6维: [x, y, z, roll, pitch, yaw] (欧拉角格式)
            duration: 移动持续时间（秒），默认3.0秒
            use_ee_control: 是否使用末端位姿控制，默认True
        
        Returns:
            bool: 是否成功执行
        """
        # 仅对A2D机器人使用此方法
        if "a2d" not in self.robot_type.lower():
            return False
        
        if not use_ee_control:
            return False
        
        if not (hasattr(self, 'robot_controller') and self.robot_controller is not None):
            return False
        
        try:
            # 步骤1: 获取当前末端位姿（使用统一方法）
            current_ee_pose = self._get_current_ee_pose()
            
            # 步骤2: 处理目标位姿格式转换
            if len(target_pose) == 7:
                # 7维格式：可能是 [x, y, z, qx, qy, qz, qw] 或 [x, y, z, rpy, gripper]
                # 检查是否是四元数格式（通常在[-1,1]范围内）还是欧拉角格式
                if np.all(np.abs(target_pose[3:6]) <= 1.0):
                    # 可能是四元数格式
                    goal_ee_pose = target_pose[:7].copy()
                else:
                    # 可能是欧拉角格式，转换为四元数
                    goal_ee_pose = np.concatenate([
                        target_pose[:3],
                        Rotation.from_euler("xyz", target_pose[3:6]).as_quat()
                    ])
            elif len(target_pose) == 6:
                # 6维格式：欧拉角格式 [x, y, z, roll, pitch, yaw]
                goal_ee_pose = np.concatenate([
                    target_pose[:3],
                    Rotation.from_euler("xyz", target_pose[3:6]).as_quat()
                ])
            else:
                print(f"Warning: Invalid target_pose dimension: {len(target_pose)}, expected 6 or 7")
                return False
            
            # 步骤3: 在生成路径前，先检查并限制目标位姿
            goal_pose_6d = np.concatenate([
                goal_ee_pose[:3],
                Rotation.from_quat(goal_ee_pose[3:]).as_euler("xyz")
            ])
            goal_pose_clipped_6d = self.clip_safety_box(goal_pose_6d)
            goal_ee_pose_clipped = np.concatenate([
                goal_pose_clipped_6d[:3],
                Rotation.from_euler("xyz", goal_pose_clipped_6d[3:]).as_quat()
            ])
            
            # 步骤4: 预先计算当前和目标位姿的欧拉角（性能优化）
            current_euler = Rotation.from_quat(current_ee_pose[3:]).as_euler("xyz")
            goal_euler = goal_pose_clipped_6d[3:]  # 已经是欧拉角，无需转换
            
            # 步骤5: 生成平滑的末端位姿路径（从当前位置到目标位置）
            cnt = int(duration / (1 / self.hz))
            if cnt < 1:
                cnt = 1
            pos_path = np.linspace(current_ee_pose[:3], goal_ee_pose_clipped[:3], cnt)
            euler_path = np.linspace(current_euler, goal_euler, cnt)
            
            # 步骤6: 使用末端位姿控制平滑移动到目标位置
            for i in range(cnt):
                target_pos = pos_path[i]
                target_euler = euler_path[i]
                
                # 应用边界限制（双重检查，确保安全）
                pose_6d = np.concatenate([target_pos, target_euler])
                pose_clipped = self.clip_safety_box(pose_6d)
                
                # 转换回四元数
                clipped_pos = pose_clipped[:3]
                clipped_quat = Rotation.from_euler("xyz", pose_clipped[3:6]).as_quat()
                
                right_pose_clipped = {
                    'x': clipped_pos[0], 'y': clipped_pos[1], 'z': clipped_pos[2],
                    'qx': clipped_quat[0], 'qy': clipped_quat[1], 'qz': clipped_quat[2], 'qw': clipped_quat[3]
                }
                
                # 使用 robot_controller 进行末端位姿控制
                lifetime = max(1.0 / self.hz, 0.1)
                self.robot_controller.set_end_effector_pose_control(
                    lifetime=lifetime,
                    control_group=['right_arm'],
                    left_pose=None,
                    right_pose=right_pose_clipped
                )
                time.sleep(1 / self.hz)
            
            return True
            
        except Exception as e:
            print(f"Warning: A2D end-effector interpolation failed: {e}")
            return False

    def _move_left_arm_to_pose_with_interpolation(
        self,
        target_pose: np.ndarray,
        duration: float = 3.0,
        use_ee_control: bool = True,
    ) -> bool:
        """
        将左臂从当前位置平滑插值移动到目标位姿。不使用范围限制（clip_safety_box）。

        Args:
            target_pose: 目标位姿，可为 7 维 [x,y,z,qx,qy,qz,qw] 或 6 维 [x,y,z,roll,pitch,yaw]
            duration: 移动时长（秒），默认 3.0
            use_ee_control: 是否用末端位姿控制，默认 True

        Returns:
            bool: 是否成功执行
        """
        if "a2d" not in self.robot_type.lower():
            return False
        if not use_ee_control:
            return False
        if not (hasattr(self, 'robot_controller') and self.robot_controller is not None):
            return False

        try:
            current_ee_pose = self._get_current_ee_pose(arm='left')
            curren_right_ee_pose = self._get_current_ee_pose(arm='right')
            right_pose = {
                    'x': curren_right_ee_pose[0], 'y': curren_right_ee_pose[1], 'z': curren_right_ee_pose[2],
                    'qx': curren_right_ee_pose[3], 'qy': curren_right_ee_pose[4], 'qz': curren_right_ee_pose[5], 'qw': curren_right_ee_pose[6]
                }
            # right_pose = {'x': 0.7367099590486855, 'y': -0.3049558173082615, 'z': 0.7104654299281462, 'qx': -0.5305025554176054, 'qy': 0.8351394817825288, 'qz': 0.07297996582241133, 'qw': 0.12563044715338886}
                
            if len(target_pose) == 7:
                if np.all(np.abs(target_pose[3:6]) <= 1.0):
                    goal_ee_pose = target_pose[:7].copy()
                else:
                    goal_ee_pose = np.concatenate([
                        target_pose[:3],
                        Rotation.from_euler("xyz", target_pose[3:6]).as_quat()
                    ])
            elif len(target_pose) == 6:
                goal_ee_pose = np.concatenate([
                    target_pose[:3],
                    Rotation.from_euler("xyz", target_pose[3:6]).as_quat()
                ])
            else:
                print(f"Warning: Invalid target_pose dimension: {len(target_pose)}, expected 6 or 7")
                return False

            current_euler = Rotation.from_quat(current_ee_pose[3:]).as_euler("xyz")
            goal_euler = Rotation.from_quat(goal_ee_pose[3:]).as_euler("xyz")

            cnt = max(1, int(duration / (1 / self.hz)))
            pos_path = np.linspace(current_ee_pose[:3], goal_ee_pose[:3], cnt)
            euler_path = np.linspace(current_euler, goal_euler, cnt)
            for i in range(cnt):
                pos = pos_path[i]
                euler = euler_path[i]
                quat = Rotation.from_euler("xyz", euler).as_quat()
                left_pose = {
                    'x': pos[0], 'y': pos[1], 'z': pos[2],
                    'qx': quat[0], 'qy': quat[1], 'qz': quat[2], 'qw': quat[3]
                }
                lifetime = max(1.0 / self.hz, 0.1)
                self.robot_controller.set_end_effector_pose_control(
                    lifetime=lifetime,
                    control_group=['left_arm', 'right_arm'],
                    left_pose=left_pose,
                    right_pose=right_pose
                )
                time.sleep(1 / self.hz)

            return True
        except Exception as e:
            print(f"Warning: A2D left arm interpolation failed: {e}")
            return False

    def _update_currpos(self, obs=None):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        if obs is None:
            obs = self._get_obs_from_robot()
        
        if "a2d" in self.robot_type.lower():
            if "arm_pose" in obs and 'single' in obs["arm_pose"]:
                self.currpos = obs["arm_pose"]['single']
            else:
                # 使用统一的获取方法（问题6：消除代码重复）
                self.currpos = self._get_current_ee_pose()
            
            if "hand_joints" in obs and 'single' in obs["hand_joints"]:
                self.curr_gripper_joints = np.array(obs["hand_joints"]['single']).squeeze()
            else:
                if hasattr(self, 'robot_station') and self.robot_station is not None:
                    try:
                        hand_states, _ = self.robot_station.hand_joint_states()
                        self.curr_gripper_joints = np.array(hand_states[6:12] if len(hand_states) > 6 else hand_states[:6]).mean()
                    except:
                        self.curr_gripper_joints = np.array([0.0])
                else:
                    self.curr_gripper_joints = np.array([0.0])
            
            if "arm_joints" in obs and 'single' in obs["arm_joints"]:
                self.curr_arm_joints = np.array(obs["arm_joints"]['single'][0:self.joint_dim])
            else:
                if hasattr(self, 'robot_station') and self.robot_station is not None:
                    try:
                        arm_joints, _ = self.robot_station.arm_joint_states()
                        self.curr_arm_joints = np.array(arm_joints[:self.joint_dim])
                    except:
                        self.curr_arm_joints = np.zeros(self.joint_dim)
                else:
                    self.curr_arm_joints = np.zeros(self.joint_dim)
        else:
            # 原有的逻辑（Franka/UR）
            self.currpos = obs["arm_pose"]['single']
            self.curr_gripper_joints = np.array(obs["hand_joints"]['single']).squeeze()
            self.curr_arm_joints = np.array(obs["arm_joints"]['single'][0:self.joint_dim])
        
        return obs

    def _get_obs(self, obs=None) -> dict:
        if self.fake_env:
            images = self.get_fake_im()
            state_observation = self.get_fake_pose()
        else:
            if obs is None:
                obs = self._get_obs_from_robot()
            self.currpos = obs["arm_pose"]['single']
            self.curr_gripper_joints = np.array(obs["hand_joints"]['single']).squeeze()
            self.curr_arm_joints = np.array(obs["arm_joints"]['single'][0:self.joint_dim])
            images = {}
            for key, cap in obs["images"].items():
                if key not in self._image_keys:
                    continue
                if "a2d" in self.robot_type.lower():
                    rgb = cap
                    images[key] = cap
                else:
                    rgb, _ = decoder_image(cap, None, bgr2rgb=self._bgr2rgb)
                    images[key] = rgb 
                if not os.path.exists(f"online_image_{key}.png"):
                    cv2.imwrite(f"online_image_{key}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            state_observation = {
                "tcp_pose": self.currpos,
                "gripper_pose": self.curr_gripper_joints,
                "joints": self.curr_arm_joints,
            }

            
        return dict(images=images, state=state_observation)

    def _get_obs_from_robot(self) -> dict:
        if "a2d" in self.robot_type.lower():
            obs = {}
            
            try:
                arm_joints, _ = self.robot_station.arm_joint_states()
                waist_joints, _ = self.robot_station.waist_joint_states()
                # hand_states, _ = self.robot_station.hand_joint_states()
                gripper_states, _ = self.robot_station.gripper_states()
                
                # 使用统一的获取方法（问题6：消除代码重复）
                arm_pose = self._get_current_ee_pose()
                
                obs['arm_joints'] = {'single': np.array(arm_joints[self.joint_dim:2*self.joint_dim])}
                obs['arm_pose'] = {'single': arm_pose}
                if len(gripper_states) > 6:
                    obs['hand_joints'] = {'single': np.array(gripper_states[1:]) / 120}
                else:
                    obs['hand_joints'] = {'single': np.array([0.0])}
                
            except Exception as e:
                print(f"Warning: A2D state retrieval failed: {e}")
                obs['arm_joints'] = {'single': np.zeros(self.joint_dim)}
                obs['arm_pose'] = {'single': np.array([0.5, 0.0, 0.8, 0, 0, 0, 1])}
                obs['hand_joints'] = {'single': np.array([0.0])}
            
            obs['images'] = {}
            if hasattr(self, 'camera_group') and self.camera_group is not None:
                try:
                    for camera_name in self.config.image_keys:
                        camera_mapping = {
                            'right': '/camera/hand_right_color',
                            'wrist': '/camera/hand_right_color',
                            'head': '/camera/head_color',
                            'left': '/camera/hand_left_color',
                        }
                        a2d_camera_name = camera_mapping.get(camera_name, f'/camera/{camera_name}_color')
                        
                        camera_image, _ = self.camera_group.get_latest_image(a2d_camera_name)
                        if camera_image is not None:
                            # target_size = self.config.image_resize.get(camera_name, [128, 128, 3])
                            # if len(target_size) == 3:
                            #     h, w = target_size[0], target_size[1]
                            # else:
                            #     h, w = 128, 128
                            
                            from PIL import Image
                            # camera_image = Image.fromarray(camera_image).resize((w, h))
                            camera_image = Image.fromarray(camera_image)
                            camera_image = np.array(camera_image)
                            obs['images'][camera_name] = camera_image
                        else:
                            target_size = self.config.image_resize.get(camera_name, [128, 128, 3])
                            h, w = target_size[0], target_size[1] if len(target_size) > 1 else 128
                            obs['images'][camera_name] = np.zeros((h, w, 3), dtype=np.uint8)
                except Exception as e:
                    print(f"Warning: A2D image retrieval failed: {e}")
                    for camera_name in self.config.image_keys:
                        target_size = self.config.image_resize.get(camera_name, [128, 128, 3])
                        h, w = target_size[0], target_size[1] if len(target_size) > 1 else 128
                        obs['images'][camera_name] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                for camera_name in self.config.image_keys:
                    target_size = self.config.image_resize.get(camera_name, [128, 128, 3])
                    h, w = target_size[0], target_size[1] if len(target_size) > 1 else 128
                    obs['images'][camera_name] = np.zeros((h, w, 3), dtype=np.uint8)
            
            return obs
        else:
            # 原有的逻辑（Franka/UR）
            obs = self.robot_station.get_obs()
            
            # standardize arm_pose
            arm_pose = obs['arm_pose']['single']
            arm_pose_t, arm_pose_quat = arm_pose[0:3], arm_pose[3:]
            arm_pose_quat = Rotation.from_quat(arm_pose_quat).as_quat(canonical=True)
            arm_pose = np.hstack([arm_pose_t, arm_pose_quat])
            obs['arm_pose'] = {'single': arm_pose}
            return obs

    def close(self):
        return

