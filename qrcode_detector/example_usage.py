#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维码检测工具使用示例
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tools.qrcode_detector import QRCodeDetector


def example_basic_detection():
    """示例1: 基本检测（相机坐标系）"""
    print("=" * 60)
    print("示例1: 基本检测（相机坐标系）")
    print("=" * 60)
    
    # 初始化检测器
    detector = QRCodeDetector(
        intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
        marker_length_mm=36.0,
        dict_name="tag36h11"
    )
    
    # 这里需要替换为实际的图片路径
    image_path = "test_image.jpg"  # 替换为你的图片路径
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  无法读取图片: {image_path}")
            print("请替换为实际的图片路径")
            return
        
        # 检测指定ID的二维码
        success, R, t, corners = detector.detect(image, target_id=4)
        
        if success:
            print(f"✅ 检测成功！Tag ID: 4")
            print(f"位置 (相机坐标系):")
            print(f"  x = {t[0]:.4f} m")
            print(f"  y = {t[1]:.4f} m")
            print(f"  z = {t[2]:.4f} m")
            print(f"\n旋转矩阵:")
            print(R)
            if corners is not None:
                print(f"\n角点坐标 (像素):")
                print(corners)
        else:
            print(f"❌ 未检测到 Tag ID: 4")
    
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_detect_all():
    """示例2: 检测所有二维码"""
    print("\n" + "=" * 60)
    print("示例2: 检测所有二维码")
    print("=" * 60)
    
    detector = QRCodeDetector(
        intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
        marker_length_mm=36.0
    )
    
    image_path = "test_image.jpg"  # 替换为你的图片路径
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  无法读取图片: {image_path}")
            return
        
        # 检测所有二维码
        results = detector.detect_all(image)
        
        if results:
            print(f"✅ 检测到 {len(results)} 个二维码:\n")
            for tag_id, (R, t, corners) in results.items():
                print(f"Tag ID: {tag_id}")
                print(f"  位置: x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
        else:
            print("❌ 未检测到任何二维码")
    
    except Exception as e:
        print(f"❌ 错误: {e}")


def example_detect_to_base():
    """示例3: 检测并转换到base坐标系"""
    print("\n" + "=" * 60)
    print("示例3: 检测并转换到base坐标系")
    print("=" * 60)
    
    detector = QRCodeDetector(
        intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
        extrinsic_path="assets/assets_wrist/hand_extrinsic_params.json",
        marker_length_mm=36.0
    )
    
    image_path = "test_image.jpg"  # 替换为你的图片路径
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  无法读取图片: {image_path}")
            return
        
        # 示例：link到base的位姿（实际使用时从机械臂获取）
        R_link2base = np.eye(3)  # 单位矩阵（示例）
        t_link2base = np.array([0.0, 0.0, 0.0])  # 零向量（示例）
        
        # 检测并转换到base坐标系
        success, R_base, t_base, corners = detector.detect_to_base(
            image,
            target_id=4,
            R_link2base=R_link2base,
            t_link2base=t_link2base
        )
        
        if success:
            print(f"✅ 检测成功！Tag ID: 4")
            print(f"位置 (base坐标系):")
            print(f"  x = {t_base[0]:.4f} m")
            print(f"  y = {t_base[1]:.4f} m")
            print(f"  z = {t_base[2]:.4f} m")
        else:
            print(f"❌ 未检测到 Tag ID: 4")
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_with_filter():
    """示例4: 使用滤波功能"""
    print("\n" + "=" * 60)
    print("示例4: 使用滤波功能")
    print("=" * 60)
    
    detector = QRCodeDetector(
        intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
        marker_length_mm=36.0
    )
    
    image_path = "test_image.jpg"  # 替换为你的图片路径
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  无法读取图片: {image_path}")
            return
        
        # 使用高斯滤波
        print("\n使用高斯滤波:")
        success, R, t, corners = detector.detect_with_filter(
            image, 
            target_id=4,
            filter_type="gaussian",
            filter_alpha=0.85
        )
        if success:
            print(f"  位置: x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
        
        # 使用SE3滤波
        print("\n使用SE3滤波:")
        sigma_obs = np.diag([
            0.005**2, 0.005**2, 0.008**2,   # xyz (m)
            (4.0*np.pi/180)**2,             # roll
            (4.0*np.pi/180)**2,             # pitch
            (8.0*np.pi/180)**2,             # yaw
        ])
        success, R, t, corners = detector.detect_with_filter(
            image,
            target_id=4,
            filter_type="se3",
            sigma_obs=sigma_obs
        )
        if success:
            print(f"  位置: x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
        
        # 重置滤波器
        detector.reset_filter()
        print("\n滤波器已重置")
    
    except Exception as e:
        print(f"❌ 错误: {e}")


def example_multiframe_filtered():
    """示例5: 多帧检测+滤波"""
    print("\n" + "=" * 60)
    print("示例5: 多帧检测+滤波")
    print("=" * 60)
    
    detector = QRCodeDetector(
        intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
        marker_length_mm=36.0
    )
    
    # 假设有多张图片
    image_paths = ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"]  # 替换为实际路径
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    if not images:
        print("⚠️  无法读取图片，请替换为实际的图片路径")
        return
    
    try:
        # 多帧检测+滤波
        success, R, t, corners = detector.detect_multiframe_filtered(
            images,
            target_id=4,
            filter_type="se3"
        )
        
        if success:
            print(f"✅ 多帧滤波成功！")
            print(f"位置: x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
            print(f"使用了 {len(images)} 帧进行滤波")
        else:
            print("❌ 多帧检测失败")
    
    except Exception as e:
        print(f"❌ 错误: {e}")


def example_with_drawing():
    """示例6: 检测并绘制结果"""
    print("\n" + "=" * 60)
    print("示例6: 检测并绘制结果")
    print("=" * 60)
    
    detector = QRCodeDetector(
        intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
        marker_length_mm=36.0
    )
    
    image_path = "test_image.jpg"  # 替换为你的图片路径
    output_path = "result_with_detection.jpg"
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  无法读取图片: {image_path}")
            return
        
        # 检测所有二维码
        results = detector.detect_all(image)
        
        # 绘制结果
        image_out = image.copy()
        for tag_id, (R, t, corners) in results.items():
            if corners is not None:
                # 绘制边界框
                cv2.polylines(image_out, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                # 绘制ID标签
                cv2.putText(image_out, f"tag{tag_id}", 
                           (corners[0][0] + 4, corners[0][1] - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # 绘制位置信息
                text = f"x:{t[0]:.3f} y:{t[1]:.3f} z:{t[2]:.3f}"
                cv2.putText(image_out, text,
                           (corners[0][0] + 4, corners[0][1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, image_out)
        print(f"✅ 结果图片已保存到: {output_path}")
        print(f"检测到 {len(results)} 个二维码")
    
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    print("二维码检测工具使用示例\n")
    print("注意: 请将示例中的图片路径替换为实际的图片路径\n")
    
    # 运行示例
    example_basic_detection()
    example_detect_all()
    example_detect_to_base()
    example_with_filter()
    example_multiframe_filtered()
    example_with_drawing()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)

