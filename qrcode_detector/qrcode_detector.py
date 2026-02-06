#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的二维码（AprilTag）检测工具
给定输入图片和相机参数，输出二维码位姿结果

使用示例：
    from tools.qrcode_detector import QRCodeDetector
    
    # 初始化检测器
    detector = QRCodeDetector(
        intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
        extrinsic_path="assets/assets_wrist/hand_extrinsic_params.json",  # 可选
        marker_length_mm=36.0,
        dict_name="tag36h11"
    )
    
    # 读取图片
    import cv2
    image = cv2.imread("test_image.jpg")
    
    # 检测二维码（返回相机坐标系下的位姿）
    success, R, t, corners = detector.detect(image, target_id=4)
    if success:
        print(f"检测成功！位置: {t}, 旋转矩阵: {R}")
    
    # 如果需要转换到base坐标系（需要提供link到base的位姿）
    R_link2base = np.eye(3)  # 示例
    t_link2base = np.array([0, 0, 0])  # 示例
    success, R_base, t_base, corners = detector.detect_to_base(
        image, target_id=4, 
        R_link2base=R_link2base, 
        t_link2base=t_link2base
    )
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
import sys

import cv2
import numpy as np

try:
    from dt_apriltags import Detector
except ImportError as e:
    print(f"❌ 导入 dt_apriltags 失败: {e}")
    print("请安装: pip install dt-apriltags")
    sys.exit(1)


# ============================================================================
# 滤波功能
# ============================================================================

def _rotm_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """旋转矩阵转四元数 (xyzw格式)"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """四元数球面线性插值（xyzw），t∈[0,1] 从 q1 到 q2，取最短路径"""
    q1 = np.array(q1, dtype=np.float64).reshape(4)
    q2 = np.array(q2, dtype=np.float64).reshape(4)
    t = float(np.clip(t, 0.0, 1.0))
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        out = (1 - t) * q1 + t * q2
    else:
        theta = np.arccos(min(1.0, dot))
        out = np.sin((1 - t) * theta) / np.sin(theta) * q1 + np.sin(t * theta) / np.sin(theta) * q2
    n = np.linalg.norm(out)
    return (out / n) if n > 0 else q1


class TagPoseFilter:
    """
    二维码位姿滤波：高斯/指数平滑滤波
    
    适用于单帧检测结果的平滑处理，减少抖动。
    """
    
    def __init__(self, gaussian_alpha: float = 0.85):
        """
        Args:
            gaussian_alpha: 高斯/指数平滑系数，越大越平滑（范围0-1）
        """
        self.gaussian_alpha = float(gaussian_alpha)
        self._pos: Optional[np.ndarray] = None
        self._quat: Optional[np.ndarray] = None
    
    def update(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入视觉观测，返回滤波后的位姿
        
        Args:
            pos: 位置 (3,)
            quat: 四元数 (4,) xyzw格式
        
        Returns:
            (position_xyz, quat_xyzw) 滤波后的位姿
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        if np.linalg.norm(quat) < 1e-8:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        quat = quat / np.linalg.norm(quat)
        
        # 高斯/指数平滑
        if self._pos is None:
            self._pos = pos.copy()
            self._quat = quat.copy()
        else:
            self._pos = self.gaussian_alpha * self._pos + (1 - self.gaussian_alpha) * pos
            self._quat = _quat_slerp(self._quat, quat, 1 - self.gaussian_alpha)
            self._quat = self._quat / np.linalg.norm(self._quat)
        
        return self._pos.copy(), self._quat.copy()
    
    def reset(self) -> None:
        """重置滤波状态"""
        self._pos = None
        self._quat = None


class TagPoseFilterSE3:
    """
    基于 SE(3) 的二维码位姿滤波（卡尔曼滤波）
    
    适用于多帧检测结果的滤波，考虑观测噪声和运动模型。
    更适合需要高精度和稳定性的场景。
    """
    
    def __init__(
        self,
        sigma_obs: np.ndarray,
        dt: float = 0.2,
        v_max: float = 0.02,
        w_max: float = 5 * np.pi / 180,
    ):
        """
        Args:
            sigma_obs: 6x6 观测噪声协方差矩阵（xyz + so3）
            dt: 时间间隔（秒）
            v_max: 最大速度（m/s）
            w_max: 最大角速度（rad/s）
        """
        drift_sigma = np.diag([
            (v_max * dt) ** 2,
            (v_max * dt) ** 2,
            (v_max * dt) ** 2,
            (w_max * dt) ** 2,
            (w_max * dt) ** 2,
            (w_max * dt) ** 2,
        ])
        self._Sigma = np.asarray(sigma_obs, dtype=np.float64)
        self._Q = np.asarray(drift_sigma, dtype=np.float64)
        
        self._T: Optional[np.ndarray] = None  # 4x4 变换矩阵
        self._P: Optional[np.ndarray] = None  # 6x6 协方差矩阵
    
    def _skew(self, v: np.ndarray) -> np.ndarray:
        """反对称矩阵"""
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])
    
    def _so3_exp(self, w: np.ndarray) -> np.ndarray:
        """SO(3) 指数映射"""
        theta = np.linalg.norm(w)
        if theta < 1e-8:
            return np.eye(3)
        k = w / theta
        K = self._skew(k)
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    def _so3_log(self, R: np.ndarray) -> np.ndarray:
        """SO(3) 对数映射"""
        cos_theta = (np.trace(R) - 1.0) * 0.5
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-8:
            return np.zeros(3)
        return (
            theta / (2.0 * np.sin(theta))
            * np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ])
        )
    
    def _se3_log(self, T: np.ndarray) -> np.ndarray:
        """SE(3) 对数映射"""
        R = T[:3, :3]
        t = T[:3, 3]
        return np.hstack([t, self._so3_log(R)])
    
    def _se3_exp(self, xi: np.ndarray) -> np.ndarray:
        """SE(3) 指数映射"""
        T = np.eye(4)
        T[:3, :3] = self._so3_exp(xi[3:])
        T[:3, 3] = xi[:3]
        return T
    
    def _pose_to_T(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """位姿转SE(3)变换矩阵"""
        x, y, z, w = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T
    
    def _T_to_pose(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SE(3)变换矩阵转位姿"""
        pos = T[:3, 3]
        R = T[:3, :3]
        qw = np.sqrt(max(0.0, 1.0 + np.trace(R))) * 0.5
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
        quat = np.array([qx, qy, qz, qw])
        return pos, quat / np.linalg.norm(quat)
    
    def update(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入观测位姿，返回滤波后的位姿
        
        Args:
            pos: 位置 (3,)
            quat: 四元数 (4,) xyzw格式
        
        Returns:
            (position_xyz, quat_xyzw) 滤波后的位姿
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        if np.linalg.norm(quat) < 1e-8:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        quat = quat / np.linalg.norm(quat)
        
        T_meas = self._pose_to_T(pos, quat)
        
        # 初始化
        if self._T is None:
            self._T = T_meas
            self._P = self._Sigma.copy()
            return pos.copy(), quat.copy()
        
        # 允许两次观测间的"未知运动"
        self._P = self._P + self._Q
        
        # SE(3) 信息滤波 update
        r = self._se3_log(np.linalg.inv(self._T) @ T_meas)
        K = self._P @ np.linalg.inv(self._P + self._Sigma)
        delta = K @ r
        
        self._T = self._T @ self._se3_exp(delta)
        self._P = (np.eye(6) - K) @ self._P
        
        return self._T_to_pose(self._T)
    
    def reset(self) -> None:
        """重置滤波状态"""
        self._T = None
        self._P = None


class QRCodeDetector:
    """
    独立的二维码（AprilTag）检测器
    
    功能：
    - 从JSON文件加载相机内参和外参
    - 图像预处理（去畸变、增强）
    - AprilTag检测和位姿解算
    - 支持相机坐标系和base坐标系输出
    """
    
    def __init__(
        self,
        intrinsic_path: Union[str, Path],
        extrinsic_path: Optional[Union[str, Path]] = None,
        marker_length_mm: float = 22.0,
        dict_name: str = "tag36h11",
        quad_decimate: float = 1.5,
        enable_enhancement: bool = True,
        enable_fallback: bool = False,
    ):
        """
        初始化检测器
        
        Args:
            intrinsic_path: 相机内参JSON文件路径
            extrinsic_path: 相机外参JSON文件路径（可选，用于转换到base坐标系）
            marker_length_mm: 二维码标记的物理尺寸（毫米）
            dict_name: AprilTag字典名称，默认"tag36h11"
            quad_decimate: 四边形检测降采样倍数，1.5平衡速度和精度
            enable_enhancement: 是否启用图像增强（CLAHE+锐化）
            enable_fallback: 是否启用备用检测器（更保守参数）
        """
        # 加载相机参数
        self.K, self.D = self._load_intrinsic(intrinsic_path)
        
        if extrinsic_path is not None:
            self.R_c2link, self.t_c2link = self._load_extrinsic(extrinsic_path)
        else:
            self.R_c2link = None
            self.t_c2link = None
        
        self.marker_length_mm = float(marker_length_mm)
        self.dict_name = dict_name
        self.enable_enhancement = enable_enhancement
        self.enable_fallback = enable_fallback
        
        # 预计算去畸变映射表（首次使用时初始化）
        self.undistort_map1 = None
        self.undistort_map2 = None
        
        # 初始化主检测器
        self.detector = Detector(
            families=dict_name,
            nthreads=1,
            quad_decimate=float(quad_decimate),
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.5,
            debug=0
        )
        
        # 初始化备用检测器（更保守参数）
        self.fallback_detector = Detector(
            families=dict_name,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        ) if enable_fallback else None
    
    @staticmethod
    def _load_intrinsic(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从JSON文件加载相机内参
        
        Args:
            path: JSON文件路径，格式包含 intrinsic.fx/fy/ppx/ppy/k1/k2/k3/p1/p2
        
        Returns:
            (K 3x3, D 1x5) 相机内参矩阵和畸变系数
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"内参文件不存在: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        intr = data["intrinsic"]
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        ppx = float(intr["ppx"])
        ppy = float(intr["ppy"])
        k1 = float(intr.get("k1", 0.0))
        k2 = float(intr.get("k2", 0.0))
        k3 = float(intr.get("k3", 0.0))
        p1 = float(intr.get("p1", 0.0))
        p2 = float(intr.get("p2", 0.0))
        
        K = np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array([[k1, k2, p1, p2, k3]], dtype=np.float64)
        
        return K, D
    
    @staticmethod
    def _load_extrinsic(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从JSON文件加载相机外参
        
        Args:
            path: JSON文件路径，格式包含 extrinsic.rotation_matrix 和 translation_vector
        
        Returns:
            (R 3x3, t 3) 相机到link的旋转矩阵和平移向量
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"外参文件不存在: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        ext = data["extrinsic"]
        R = np.array(ext["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        t = np.array(ext["translation_vector"], dtype=np.float64).reshape(3)
        
        return R, t
    
    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        去图像畸变（预计算映射表，重复使用以加速）
        
        Args:
            image: 输入图像（灰度或彩色）
        
        Returns:
            去畸变后的图像
        """
        if self.undistort_map1 is None or self.undistort_map2 is None:
            h, w = image.shape[:2]
            self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K, (w, h), cv2.CV_16SC2
            )
        return cv2.remap(image, self.undistort_map1, self.undistort_map2, cv2.INTER_LINEAR)
    
    def _enhance_image(self, gray: np.ndarray) -> np.ndarray:
        """
        图像增强：CLAHE（对比度受限自适应直方图均衡化）+ 锐化
        
        Args:
            gray: 输入灰度图
        
        Returns:
            增强后的图像
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced
    
    @staticmethod
    def _compose_RT(R2: np.ndarray, t2: np.ndarray, R1: np.ndarray, t1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        组合变换: A->C = (A->B) * (B->C)
        
        Args:
            R2, t2: 变换 B->C
            R1, t1: 变换 A->B
        
        Returns:
            (R, t) 变换 A->C
        """
        R = R2 @ R1
        t = (R2 @ t1.reshape(3, 1)).reshape(3) + t2.reshape(3)
        return R, t
    
    def detect(
        self,
        image: np.ndarray,
        target_id: int,
        return_corners: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测二维码，返回相机坐标系下的位姿
        
        Args:
            image: 输入BGR图像（numpy数组）
            target_id: 目标二维码ID
            return_corners: 是否返回角点坐标
        
        Returns:
            (success, R, t, corners)
            - success: 是否检测成功
            - R: 3x3旋转矩阵（二维码到相机）
            - t: 3x1平移向量（二维码到相机，单位：米）
            - corners: 4x2角点坐标（像素坐标），如果return_corners=False则为None
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 去畸变
            gray_undistorted = self._undistort_image(gray)
            
            # 准备检测参数
            camera_params = (
                float(self.K[0, 0]),  # fx
                float(self.K[1, 1]),  # fy
                float(self.K[0, 2]),  # cx
                float(self.K[1, 2]),  # cy
            )
            marker_size_m = self.marker_length_mm / 1000.0
            
            # 三级检测策略
            tags = None
            
            # 策略1: 增强图 + 主检测器
            if self.enable_enhancement:
                gray_enhanced = self._enhance_image(gray_undistorted)
                tags = self.detector.detect(gray_enhanced, True, camera_params, marker_size_m)
                if tags is None or target_id not in [int(t.tag_id) for t in tags]:
                    tags = None
            
            # 策略2: 原图 + 主检测器
            if tags is None:
                tags = self.detector.detect(gray_undistorted, True, camera_params, marker_size_m)
            
            # 策略3: 备用检测器（增强图 -> 原图）
            if (tags is None or target_id not in [int(t.tag_id) for t in tags]) and self.enable_fallback and self.fallback_detector is not None:
                if self.enable_enhancement:
                    gray_enhanced = self._enhance_image(gray_undistorted)
                    tags = self.fallback_detector.detect(gray_enhanced, True, camera_params, marker_size_m)
                if tags is None or len(tags) == 0:
                    tags = self.fallback_detector.detect(gray_undistorted, True, camera_params, marker_size_m)
            
            # 检查是否检测到目标
            if tags is None or len(tags) == 0:
                return False, None, None, None
            
            # 查找目标ID
            apriltag_R = None
            apriltag_t = None
            corners = None
            
            for tag in tags:
                if int(tag.tag_id) == int(target_id):
                    apriltag_R = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
                    apriltag_t = np.asarray(tag.pose_t, dtype=np.float64).reshape(3)
                    if return_corners:
                        corners = tag.corners.astype(int)
                    break
            
            if apriltag_R is None or apriltag_t is None:
                return False, None, None, None
            
            # PnP解歧义：选择tag法向朝向相机的解（tag Z在相机系Z分量应 < 0）
            z_axis_in_cam = apriltag_R[:, 2]
            if z_axis_in_cam[2] > 0:
                R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
                apriltag_R = apriltag_R @ R_flip
            
            return True, apriltag_R, apriltag_t, corners
        
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None, None
    
    def detect_to_base(
        self,
        image: np.ndarray,
        target_id: int,
        R_link2base: np.ndarray,
        t_link2base: np.ndarray,
        return_corners: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测二维码，返回base坐标系下的位姿
        
        Args:
            image: 输入BGR图像
            target_id: 目标二维码ID
            R_link2base: 3x3旋转矩阵（link到base）
            t_link2base: 3x1平移向量（link到base，单位：米）
            return_corners: 是否返回角点坐标
        
        Returns:
            (success, R_base, t_base, corners)
            - success: 是否检测成功
            - R_base: 3x3旋转矩阵（二维码到base）
            - t_base: 3x1平移向量（二维码到base，单位：米）
            - corners: 4x2角点坐标（像素坐标），如果return_corners=False则为None
        """
        if self.R_c2link is None or self.t_c2link is None:
            raise ValueError("未提供外参文件，无法转换到base坐标系。请使用detect()方法获取相机坐标系下的位姿。")
        
        # 先检测相机坐标系下的位姿
        success, R_tag2cam, t_tag2cam, corners = self.detect(image, target_id, return_corners)
        
        if not success:
            return False, None, None, None
        
        # 坐标变换链: tag -> camera -> link -> base
        R_cam2link = self.R_c2link
        t_cam2link = self.t_c2link
        
        R_link2base = np.asarray(R_link2base, dtype=np.float64).reshape(3, 3)
        t_link2base = np.asarray(t_link2base, dtype=np.float64).reshape(3)
        
        # camera -> link -> base
        R_cam2base, t_cam2base = self._compose_RT(R_link2base, t_link2base, R_cam2link, t_cam2link)
        
        # tag -> camera -> base
        R_tag2base, t_tag2base = self._compose_RT(R_cam2base, t_cam2base, R_tag2cam, t_tag2cam)
        
        return True, R_tag2base, t_tag2base, corners
    
    def detect_all(
        self,
        image: np.ndarray,
        return_corners: bool = True,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        检测图像中所有二维码
        
        Args:
            image: 输入BGR图像
            return_corners: 是否返回角点坐标
        
        Returns:
            {tag_id: (R, t, corners)} 字典
            - tag_id: 二维码ID
            - R: 3x3旋转矩阵（二维码到相机）
            - t: 3x1平移向量（二维码到相机，单位：米）
            - corners: 4x2角点坐标（像素坐标），如果return_corners=False则为None
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 去畸变
            gray_undistorted = self._undistort_image(gray)
            
            # 准备检测参数
            camera_params = (
                float(self.K[0, 0]),
                float(self.K[1, 1]),
                float(self.K[0, 2]),
                float(self.K[1, 2]),
            )
            marker_size_m = self.marker_length_mm / 1000.0
            
            # 检测所有二维码
            tags = self.detector.detect(gray_undistorted, True, camera_params, marker_size_m)
            
            if tags is None or len(tags) == 0:
                return {}
            
            result = {}
            for tag in tags:
                tag_id = int(tag.tag_id)
                apriltag_R = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
                apriltag_t = np.asarray(tag.pose_t, dtype=np.float64).reshape(3)
                
                # PnP解歧义
                z_axis_in_cam = apriltag_R[:, 2]
                if z_axis_in_cam[2] > 0:
                    R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
                    apriltag_R = apriltag_R @ R_flip
                
                corners = tag.corners.astype(int) if return_corners else None
                result[tag_id] = (apriltag_R, apriltag_t, corners)
            
            return result
        
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _quat_to_rotm(self, quat: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵 (xyzw格式)"""
        x, y, z, w = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)
        return R
    
    def detect_with_filter(
        self,
        image: np.ndarray,
        target_id: int,
        filter_type: str = "gaussian",
        filter_alpha: float = 0.85,
        sigma_obs: Optional[np.ndarray] = None,
        return_corners: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测二维码并使用滤波器平滑结果
        
        Args:
            image: 输入BGR图像
            target_id: 目标二维码ID
            filter_type: 滤波器类型，"gaussian" 或 "se3"
            filter_alpha: 高斯滤波器的平滑系数（仅用于gaussian类型）
            sigma_obs: SE3滤波器的观测噪声协方差矩阵（仅用于se3类型）
            return_corners: 是否返回角点坐标
        
        Returns:
            (success, R, t, corners) 滤波后的位姿
        """
        # 先检测
        success, R, t, corners = self.detect(image, target_id, return_corners)
        
        if not success:
            return False, None, None, None
        
        # 转换为四元数
        quat = _rotm_to_quat_xyzw(R)
        
        # 应用滤波
        if filter_type == "gaussian":
            if not hasattr(self, '_gaussian_filter') or self._gaussian_filter is None:
                self._gaussian_filter = TagPoseFilter(gaussian_alpha=filter_alpha)
            pos_filtered, quat_filtered = self._gaussian_filter.update(t, quat)
            # 四元数转回旋转矩阵
            R_filtered = self._quat_to_rotm(quat_filtered)
            return True, R_filtered, pos_filtered, corners
        
        elif filter_type == "se3":
            if sigma_obs is None:
                # 默认观测噪声协方差
                sigma_obs = np.diag([
                    0.005**2, 0.005**2, 0.008**2,   # xyz (m)
                    (4.0*np.pi/180)**2,             # roll
                    (4.0*np.pi/180)**2,             # pitch
                    (8.0*np.pi/180)**2,             # yaw
                ])
            if not hasattr(self, '_se3_filter') or self._se3_filter is None:
                self._se3_filter = TagPoseFilterSE3(sigma_obs=sigma_obs)
            pos_filtered, quat_filtered = self._se3_filter.update(t, quat)
            # 四元数转回旋转矩阵
            R_filtered = self._quat_to_rotm(quat_filtered)
            return True, R_filtered, pos_filtered, corners
        
        else:
            raise ValueError(f"未知的滤波器类型: {filter_type}，支持 'gaussian' 或 'se3'")
    
    def detect_multiframe_filtered(
        self,
        images: list,
        target_id: int,
        filter_type: str = "se3",
        sigma_obs: Optional[np.ndarray] = None,
        return_corners: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        多帧检测+滤波：对多张图片进行检测，使用滤波器平滑结果
        
        适用于需要稳定位姿的场景，通过多帧滤波减少单帧误差。
        
        Args:
            images: 输入BGR图像列表
            target_id: 目标二维码ID
            filter_type: 滤波器类型，"gaussian" 或 "se3"（推荐se3）
            sigma_obs: SE3滤波器的观测噪声协方差矩阵
            return_corners: 是否返回角点坐标（返回最后一帧的角点）
        
        Returns:
            (success, R, t, corners) 滤波后的位姿（最后一帧的滤波结果）
        """
        if sigma_obs is None:
            sigma_obs = np.diag([
                0.005**2, 0.005**2, 0.008**2,   # xyz (m)
                (4.0*np.pi/180)**2,             # roll
                (4.0*np.pi/180)**2,             # pitch
                (8.0*np.pi/180)**2,             # yaw
            ])
        
        # 初始化滤波器
        if filter_type == "gaussian":
            filter_obj = TagPoseFilter(gaussian_alpha=0.85)
        elif filter_type == "se3":
            filter_obj = TagPoseFilterSE3(sigma_obs=sigma_obs)
        else:
            raise ValueError(f"未知的滤波器类型: {filter_type}")
        
        last_corners = None
        success_count = 0
        
        # 逐帧检测并滤波
        for image in images:
            success, R, t, corners = self.detect(image, target_id, return_corners)
            
            if success:
                success_count += 1
                quat = _rotm_to_quat_xyzw(R)
                pos_filtered, quat_filtered = filter_obj.update(t, quat)
                last_corners = corners
            else:
                # 如果某帧检测失败，跳过该帧（滤波器保持之前的状态）
                continue
        
        if success_count == 0:
            return False, None, None, None
        
        # 获取最后一帧的滤波结果
        if filter_type == "gaussian":
            pos_final, quat_final = filter_obj._pos, filter_obj._quat
        else:  # se3
            pos_final, quat_final = filter_obj._T_to_pose(filter_obj._T)
        
        R_final = self._quat_to_rotm(quat_final)
        
        return True, R_final, pos_final, last_corners
    
    def reset_filter(self, filter_type: str = "all") -> None:
        """
        重置滤波器状态
        
        Args:
            filter_type: 要重置的滤波器类型，"gaussian", "se3", 或 "all"
        """
        if filter_type in ["gaussian", "all"]:
            if hasattr(self, '_gaussian_filter') and self._gaussian_filter is not None:
                self._gaussian_filter.reset()
        
        if filter_type in ["se3", "all"]:
            if hasattr(self, '_se3_filter') and self._se3_filter is not None:
                self._se3_filter.reset()


def main():
    """命令行测试示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="二维码检测工具测试")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--intrinsic", type=str, required=True, help="相机内参JSON文件路径")
    parser.add_argument("--extrinsic", type=str, default=None, help="相机外参JSON文件路径（可选）")
    parser.add_argument("--tag-id", type=int, default=None, help="目标二维码ID（可选，不指定则检测所有）")
    parser.add_argument("--marker-length-mm", type=float, default=36.0, help="二维码物理尺寸（毫米）")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径（可选，绘制检测结果）")
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = QRCodeDetector(
        intrinsic_path=args.intrinsic,
        extrinsic_path=args.extrinsic,
        marker_length_mm=args.marker_length_mm,
    )
    
    # 读取图片
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ 无法读取图片: {args.image}")
        return 1
    
    print(f"图片尺寸: {image.shape}")
    
    # 检测
    if args.tag_id is not None:
        # 检测指定ID
        success, R, t, corners = detector.detect(image, args.tag_id)
        if success:
            print(f"\n✅ 检测成功！Tag ID: {args.tag_id}")
            print(f"位置 (相机坐标系): x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
            print(f"旋转矩阵:\n{R}")
            if corners is not None:
                print(f"角点坐标 (像素):\n{corners}")
        else:
            print(f"\n❌ 未检测到 Tag ID: {args.tag_id}")
    else:
        # 检测所有
        results = detector.detect_all(image)
        if results:
            print(f"\n✅ 检测到 {len(results)} 个二维码:")
            for tag_id, (R, t, corners) in results.items():
                print(f"\nTag ID: {tag_id}")
                print(f"位置 (相机坐标系): x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
        else:
            print("\n❌ 未检测到任何二维码")
    
    # 绘制结果（如果指定输出路径）
    if args.output:
        image_out = image.copy()
        if args.tag_id is not None and success:
            # 绘制指定ID的检测结果
            if corners is not None:
                cv2.polylines(image_out, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(image_out, f"tag{args.tag_id}", 
                           (corners[0][0] + 4, corners[0][1] - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # 绘制所有检测结果
            results = detector.detect_all(image)
            for tag_id, (R, t, corners) in results.items():
                if corners is not None:
                    cv2.polylines(image_out, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(image_out, f"tag{tag_id}", 
                               (corners[0][0] + 4, corners[0][1] - 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(args.output, image_out)
        print(f"\n结果图片已保存到: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())

