import time
import numpy as np
from gymnasium import Env, spaces
import gymnasium as gym
from scipy.spatial.transform import Rotation
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space, flatten
# xRocs logger (only needed for Franka/UR robots)
try:
    from xrocs.utils.logger.logger_loader import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from rl_envs.shared_state import shared_state
import cv2
import traceback
import sys

class HumanIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type
        self.env.unwrapped.init_xtele() # init xtele
        self.control_mode = env.unwrapped.control_mode
        self.enable_rotation = env.unwrapped.enable_rotation
    

    def reset(self, **kwargs):
        """Reset the environment and sync robot position."""
        obs, info = self.env.reset(**kwargs)
        shared_state.human_intervention_key = False
        self.env.unwrapped.sync_xtele(timeout=2)
        info["is_intervention"] = False
        return obs, info

    def pose2matrix(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = Rotation.from_quat(pose_quat).as_matrix()
        pose_matrix[:3, 3] = pose_t
        return pose_matrix
    


    def transform_pose(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        curr_matrix = self.pose2matrix(current_pose)
        tar_matrix = self.pose2matrix(target_pose)


        T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)

        return T_diff_matrix

    def action(self, action: np.ndarray) -> np.ndarray:
        # intervened = True
        intervened = shared_state.human_intervention_key
        if intervened:
            try:
                obs = self.env.unwrapped.get_xtele()
                xtele_joints, xtele_pose = obs['joints'], obs['pose']
                
                # 对于 a2d 机器人，get_xtele() 返回的是增量向量
                if 'a2d' in self.robot_type:
                    # pose_delta 是 7 维向量 [delta_pos(3) + delta_rot(3) + gripper_delta(1)]
                    pose_delta = np.array(xtele_pose)  # 或 xtele_joints，两者相同
                    
                    # 解析增量向量
                    delta_pos = pose_delta[:3]  # 位置增量
                    delta_rot = pose_delta[3:6]  # 旋转增量（欧拉角）
                    gripper_delta = pose_delta[6]  # 夹爪增量
                    
                    # 直接使用增量向量，归一化后作为动作
                    expert_a = np.zeros(7, dtype=np.float32)
                    expert_a[:3] = delta_pos
                    expert_a[3:6] = delta_rot
                    expert_a[6:] = gripper_delta
                    
                    # 边缘裁剪
                    epsilon = 1e-6
                    expert_a[0:6] = expert_a[0:6].clip(-1+epsilon, 1-epsilon)
                    
                    return expert_a, pose_delta, True
                else:
                    # 对于其他机器人（franka/ur），使用原有逻辑
                    # print("gripper_value:", len(xtele_joints), xtele_joints[-1])
                    if self.control_mode == "joint":
                        expert_a = xtele_joints
                    else:
                        curr_matrix = self.pose2matrix(self.env.unwrapped.currpos)
                        tar_matrix = self.pose2matrix(xtele_pose)
                        T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)

                        
                        rel_rot = Rotation.from_matrix(T_diff_matrix[:3, :3]).as_euler("xyz")
                        rel_pos = T_diff_matrix[:3, 3]
                        expert_a = np.zeros(7, dtype=np.float32)
                        expert_a[:3] = rel_pos / self.env.unwrapped.action_scale[0]
                        expert_a[3:6] = rel_rot / self.env.unwrapped.action_scale[1]
                        expert_a[6:] = xtele_joints[-1] / self.env.unwrapped.action_scale[2]
                        
                        # expert_a = np.clip(expert_a, [-1]*7, [1]*7)
                        """
                        intervention action 边缘裁剪
                        """
                        epsilon = 1e-6
                        expert_a[0:6]= expert_a[0:6].clip(-1+epsilon, 1-epsilon)

                    return expert_a, xtele_joints, True
            except Exception as e:
                print(f"Error in action: {e}")
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()          # full stacktrace
                sys.exit(1)
        return action, None, False

    def step(self, action):
        action, xtele_joints,replaced = self.action(action)
        if replaced:
            obs, rew, terminated, truncated, info = self.env.step(action)
            info["intervene_action"] = action
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)        
            self.env.unwrapped.sync_xtele(timeout=0.1)
        

        info["is_intervention"] = replaced
        return obs, rew, terminated, truncated, info


class SpaceMouseIntervention(gym.ActionWrapper):
    """
    SpaceMouse干预包装器，直接使用pyspacemouse读取输入，避免嵌套调用导致的延迟。
    直接读取SpaceMouse的增量值作为动作，不需要考虑joints。
    """
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type
        self.control_mode = env.unwrapped.control_mode
        self.enable_rotation = env.unwrapped.enable_rotation
        # 初始化时从环境获取当前夹爪状态，而不是总是从 0 开始
        if hasattr(env.unwrapped, 'last_gripper_value'):
            self.pre_button_state = env.unwrapped.last_gripper_value
        else:
            self.pre_button_state = 0

    def read_latest(self):
        """
        读取最新状态，减少缓冲区读取次数以降低延迟
        """
        latest_state = None
        for _ in range(3):
            state = self.pyspacemouse.read()
            if state is not None:
                latest_state = state
        return latest_state

    def get_delta(self):
        """
        读取SpaceMouse的增量值（阻塞式，持续等待直到读取到有效输入）
        
        返回: dict with 'delta_pos', 'delta_rot', 'gripper_delta'
        """

        # while True:
        import pyspacemouse
        try:
            device =  pyspacemouse.open()
            while True:
                state = device.read()
                
                # 提取增量（调整坐标系：y 和 z 取反）
                delta_pos = np.array([state.x, -state.y, -state.z])
                delta_rot = np.array([state.pitch, state.roll, state.yaw])
                
                # 提取按钮状态
                buttons = list(state.buttons) if state.buttons else []
                if len(buttons) > 0 and (buttons[0] == 1 or buttons[1] == 1):
                    gripper_delta = 1.0
                else:
                    gripper_delta = 0.0
                has_button = any(b == 1 for b in buttons) if buttons else False
                if has_button:
                    self.pre_button_state = 1 - self.pre_button_state
                # 检查是否有有效输入（降低阈值提高灵敏度）
                has_movement = (np.abs(delta_pos).sum() > 0.1 or 
                            np.abs(delta_rot).sum() > 0.1)
                state_dict = {
                    'delta_pos': delta_pos,
                    'delta_rot': delta_rot,
                    'gripper_delta': self.pre_button_state
                }
                # print("state:", state_dict, "has_movement:", has_movement, "has_button:", has_button)
                # # 如果有有效输入，立即返回
                if has_movement or has_button:
                    has_movement = 0
                    has_button = 0
                    break

                # 如果没有有效输入，短暂等待后继续读取
                time.sleep(0.001)
        except Exception as e:
            print(f"Error in SpaceMouseIntervention.get_delta: {e}")
            print(f"[{type(e).__name__}] {e!r}")
            traceback.print_exc()
            # sys.exit(1)
            return None
        finally:
            pyspacemouse.close()
            return state_dict

    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        shared_state.human_intervention_key = False
        info["is_intervention"] = False
        return obs, info

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        检查是否需要干预，如果需要则直接从SpaceMouse读取增量作为动作
        """
        intervened = shared_state.human_intervention_key
        if intervened:
            try:
                # 在切换到人工干预时，同步当前夹爪状态
                # 避免切换时夹爪状态被重置为 0
                if hasattr(self.env.unwrapped, 'last_gripper_value'):
                    self.pre_button_state = self.env.unwrapped.last_gripper_value
                
                # 直接从SpaceMouse读取增量（阻塞式，等待有效输入）
                delta = self.get_delta()
                
                # 直接使用增量向量作为动作
                # 格式: [delta_pos(3), delta_rot(3), gripper_delta(1)]
                expert_a = np.zeros(7, dtype=np.float32)
                expert_a[:3] = delta['delta_pos']
                expert_a[3:6] = delta['delta_rot']
                expert_a[6] = delta['gripper_delta']
                
                # 边缘裁剪
                epsilon = 1e-6
                expert_a[0:6] = expert_a[0:6].clip(-1+epsilon, 1-epsilon)
                
                # 构建pose_delta用于返回（与HumanIntervention接口保持一致）
                pose_delta = np.concatenate([
                    delta['delta_pos'],
                    delta['delta_rot'],
                    [delta['gripper_delta']]
                ])
                
                return expert_a, pose_delta, True
            except Exception as e:
                print(f"Error in SpaceMouseIntervention.action: {e}")
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()
                sys.exit(1)
        
        return action, None, False

    def step(self, action):
        """执行步骤，如果被干预则使用SpaceMouse输入，否则使用原动作"""
        action, pose_delta, replaced = self.action(action)
        if replaced:
            obs, rew, terminated, truncated, info = self.env.step(action)
            info["intervene_action"] = action
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)
        
        info["is_intervention"] = replaced
        return obs, rew, terminated, truncated, info

    def close(self):
        """关闭SpaceMouse设备"""
        try:
            if hasattr(self, 'pyspacemouse'):
                self.pyspacemouse.close()
        except:
            pass

    def __del__(self):
        """析构时关闭设备"""
        self.close()


class AugmentedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env

    def observation(self, obs):
        images = obs['images']
        env = self.env.unwrapped
        for key, img in images.items():
            if hasattr(env, 'image_crop'):
                cropped_rgb = env.image_crop[key](img) if key in env.image_crop else img
            else:
                cropped_rgb = img
            cropped_rgb = cv2.resize(
                cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = cropped_rgb

        return obs
    
    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info






class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], Rotation.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )


        return observation


from collections import OrderedDict


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None, use_force=False):
        super().__init__(env)
        if use_force:
            self.proprio_keys = proprio_keys
        else:
            self.proprio_keys = proprio_keys[:2]

        print("proprio_keys:", self.proprio_keys)    

        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            OrderedDict((key, self.env.observation_space["state"][key]) for key in self.proprio_keys)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        from collections import OrderedDict
        obs = {
            "state": flatten(
                self.proprio_space,
                OrderedDict((key, obs["state"][key]) for key in self.proprio_keys),
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

  
def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs