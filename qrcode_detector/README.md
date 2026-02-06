# 二维码检测工具

独立的二维码（AprilTag）检测工具，可以方便地集成到其他项目中。

## 功能特性

- ✅ 独立的检测模块，最小化外部依赖
- ✅ 支持从JSON文件加载相机内参和外参
- ✅ 三级检测策略，提高检测成功率
- ✅ 图像预处理（去畸变、CLAHE增强、锐化）
- ✅ 支持相机坐标系和base坐标系输出
- ✅ 支持检测单个或多个二维码
- ✅ **内置位姿滤波功能**（高斯平滑和SE3卡尔曼滤波）
- ✅ **多帧检测+滤波**，提高位姿稳定性

## 依赖

```bash
pip install opencv-python numpy dt-apriltags
```

## 使用方法

### 1. Python API 使用

#### 基本使用（相机坐标系）

```python
from tools.qrcode_detector import QRCodeDetector
import cv2

# 初始化检测器
detector = QRCodeDetector(
    intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
    marker_length_mm=36.0,
    dict_name="tag36h11"
)

# 读取图片
image = cv2.imread("test_image.jpg")

# 检测指定ID的二维码
success, R, t, corners = detector.detect(image, target_id=4)

if success:
    print(f"检测成功！")
    print(f"位置 (相机坐标系): x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
    print(f"旋转矩阵:\n{R}")
    print(f"角点坐标:\n{corners}")
else:
    print("未检测到二维码")
```

#### 转换到base坐标系

```python
import numpy as np
from tools.qrcode_detector import QRCodeDetector

# 初始化检测器（需要提供外参）
detector = QRCodeDetector(
    intrinsic_path="assets/assets_wrist/hand_intrinsic_params.json",
    extrinsic_path="assets/assets_wrist/hand_extrinsic_params.json",
    marker_length_mm=36.0
)

# 读取图片
image = cv2.imread("test_image.jpg")

# 提供link到base的位姿（例如从机械臂获取）
R_link2base = np.eye(3)  # 示例：单位矩阵
t_link2base = np.array([0.0, 0.0, 0.0])  # 示例：零向量

# 检测并转换到base坐标系
success, R_base, t_base, corners = detector.detect_to_base(
    image, 
    target_id=4,
    R_link2base=R_link2base,
    t_link2base=t_link2base
)

if success:
    print(f"位置 (base坐标系): x={t_base[0]:.4f}m, y={t_base[1]:.4f}m, z={t_base[2]:.4f}m")
```

#### 检测所有二维码

```python
# 检测图像中所有二维码
results = detector.detect_all(image)

for tag_id, (R, t, corners) in results.items():
    print(f"Tag ID: {tag_id}")
    print(f"位置: x={t[0]:.4f}m, y={t[1]:.4f}m, z={t[2]:.4f}m")
```

#### 使用滤波功能

```python
# 单帧检测+滤波（高斯平滑）
success, R, t, corners = detector.detect_with_filter(
    image, 
    target_id=4,
    filter_type="gaussian",
    filter_alpha=0.85  # 平滑系数，越大越平滑
)

# 单帧检测+滤波（SE3卡尔曼滤波，推荐）
import numpy as np
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

# 多帧检测+滤波（推荐用于高精度场景）
images = [image1, image2, image3, image4, image5]  # 多张图片
success, R, t, corners = detector.detect_multiframe_filtered(
    images,
    target_id=4,
    filter_type="se3"  # 或 "gaussian"
)

# 重置滤波器状态（切换检测目标时）
detector.reset_filter(filter_type="all")
```

### 2. 命令行使用

```bash
# 检测指定ID的二维码
python tools/qrcode_detector.py \
    --image test_image.jpg \
    --intrinsic assets/assets_wrist/hand_intrinsic_params.json \
    --tag-id 4 \
    --marker-length-mm 36.0 \
    --output result.jpg

# 检测所有二维码
python tools/qrcode_detector.py \
    --image test_image.jpg \
    --intrinsic assets/assets_wrist/hand_intrinsic_params.json \
    --marker-length-mm 36.0 \
    --output result.jpg
```

## API 参考

### QRCodeDetector 类

#### `__init__()`

初始化检测器

**参数：**
- `intrinsic_path`: 相机内参JSON文件路径（必需）
- `extrinsic_path`: 相机外参JSON文件路径（可选）
- `marker_length_mm`: 二维码物理尺寸（毫米），默认22.0
- `dict_name`: AprilTag字典名称，默认"tag36h11"
- `quad_decimate`: 四边形检测降采样倍数，默认1.5
- `enable_enhancement`: 是否启用图像增强，默认True
- `enable_fallback`: 是否启用备用检测器，默认False

#### `detect(image, target_id, return_corners=True)`

检测指定ID的二维码，返回相机坐标系下的位姿

**参数：**
- `image`: 输入BGR图像（numpy数组）
- `target_id`: 目标二维码ID
- `return_corners`: 是否返回角点坐标

**返回：**
- `(success, R, t, corners)`
  - `success`: 是否检测成功（bool）
  - `R`: 3x3旋转矩阵（二维码到相机）
  - `t`: 3x1平移向量（二维码到相机，单位：米）
  - `corners`: 4x2角点坐标（像素坐标），如果return_corners=False则为None

#### `detect_to_base(image, target_id, R_link2base, t_link2base, return_corners=True)`

检测二维码并转换到base坐标系

**参数：**
- `image`: 输入BGR图像
- `target_id`: 目标二维码ID
- `R_link2base`: 3x3旋转矩阵（link到base）
- `t_link2base`: 3x1平移向量（link到base，单位：米）
- `return_corners`: 是否返回角点坐标

**返回：**
- `(success, R_base, t_base, corners)`
  - `success`: 是否检测成功
  - `R_base`: 3x3旋转矩阵（二维码到base）
  - `t_base`: 3x1平移向量（二维码到base，单位：米）
  - `corners`: 4x2角点坐标（像素坐标）

#### `detect_all(image, return_corners=True)`

检测图像中所有二维码

**参数：**
- `image`: 输入BGR图像
- `return_corners`: 是否返回角点坐标

**返回：**
- `{tag_id: (R, t, corners)}` 字典

#### `detect_with_filter(image, target_id, filter_type="gaussian", filter_alpha=0.85, sigma_obs=None, return_corners=True)`

检测二维码并使用滤波器平滑结果

**参数：**
- `image`: 输入BGR图像
- `target_id`: 目标二维码ID
- `filter_type`: 滤波器类型，"gaussian" 或 "se3"
- `filter_alpha`: 高斯滤波器的平滑系数（0-1，仅用于gaussian类型）
- `sigma_obs`: SE3滤波器的观测噪声协方差矩阵6x6（仅用于se3类型）
- `return_corners`: 是否返回角点坐标

**返回：**
- `(success, R, t, corners)` 滤波后的位姿

#### `detect_multiframe_filtered(images, target_id, filter_type="se3", sigma_obs=None, return_corners=True)`

多帧检测+滤波：对多张图片进行检测，使用滤波器平滑结果

**参数：**
- `images`: 输入BGR图像列表
- `target_id`: 目标二维码ID
- `filter_type`: 滤波器类型，"gaussian" 或 "se3"（推荐se3）
- `sigma_obs`: SE3滤波器的观测噪声协方差矩阵6x6
- `return_corners`: 是否返回角点坐标（返回最后一帧的角点）

**返回：**
- `(success, R, t, corners)` 滤波后的位姿（最后一帧的滤波结果）

#### `reset_filter(filter_type="all")`

重置滤波器状态

**参数：**
- `filter_type`: 要重置的滤波器类型，"gaussian", "se3", 或 "all"

## 相机参数文件格式

### 内参文件格式 (intrinsic_path)

```json
{
    "intrinsic": {
        "fx": 613.6268372564538,
        "fy": 613.7904368057748,
        "ppx": 641.4782222342511,
        "ppy": 403.87122963900003,
        "k1": -0.033529918354863256,
        "k2": 0.061559726719317275,
        "k3": -0.04084132180881275,
        "p1": -0.0008961934964638241,
        "p2": 0.00014004676435436423
    }
}
```

### 外参文件格式 (extrinsic_path)

```json
{
    "extrinsic": {
        "rotation_matrix": [
            [0.087, -0.539, -0.838],
            [0.093, 0.842, -0.532],
            [0.992, -0.032, 0.124]
        ],
        "translation_vector": [
            -0.047,
            0.041,
            -0.063
        ]
    }
}
```

## 检测策略

工具使用三级检测策略以提高检测成功率：

1. **策略1**: 增强图（CLAHE+锐化）+ 主检测器
2. **策略2**: 原图 + 主检测器
3. **策略3**: 备用检测器（更保守参数，增强图 → 原图）

如果策略1失败，自动尝试策略2；如果仍未检测到目标ID，且启用了备用检测器，则尝试策略3。

## 滤波功能说明

工具内置两种滤波器，用于平滑位姿检测结果，减少抖动：

### 1. TagPoseFilter（高斯平滑）

- **原理**：指数加权移动平均（EWMA）
- **适用场景**：单帧检测结果的实时平滑
- **参数**：`gaussian_alpha`（0-1），越大越平滑，默认0.85
- **优点**：计算简单，延迟低
- **缺点**：不考虑观测噪声模型

### 2. TagPoseFilterSE3（SE3卡尔曼滤波）

- **原理**：基于SE(3)流形的卡尔曼滤波
- **适用场景**：多帧检测的高精度滤波（推荐）
- **参数**：`sigma_obs`（6x6观测噪声协方差矩阵）
- **优点**：考虑观测噪声，精度高，适合多帧滤波
- **缺点**：计算稍复杂

### 使用建议

- **单帧实时检测**：使用 `detect_with_filter(filter_type="gaussian")`
- **多帧高精度检测**：使用 `detect_multiframe_filtered(filter_type="se3")`
- **切换检测目标**：调用 `reset_filter()` 重置滤波器状态

## 注意事项

1. 确保相机内参准确，否则位姿解算会有误差
2. 二维码物理尺寸（`marker_length_mm`）必须与实际尺寸一致
3. 如果只使用相机坐标系，可以不提供外参文件
4. 转换到base坐标系需要提供link到base的位姿（通常从机械臂获取）
5. 滤波功能会保持状态，切换检测目标时记得调用 `reset_filter()`
6. 多帧滤波时，如果某帧检测失败，滤波器会保持之前的状态继续工作

