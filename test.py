"""
完整的计算机视觉测距作业脚本

本脚本实现了从图像中自动检测标定点和人体，计算单应性矩阵，
并将人体脚部坐标从像素平面转换到世界坐标系，
最后计算脚部与电网设备区域的最短欧氏距离。

请确保已安装所需库:
pip install ultralytics opencv-python numpy shapely
"""

import cv2
import numpy as np
import json
import glob
import re  # 用于从文件名中提取ID
import os
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# --- 1. 配置路径 (!!! 请根据你的文件位置修改以下路径 !!!) ---

# 你训练好的标定点检测模型
DOT_MODEL_PATH = 'runs/detect/dot_detector3/weights/best.pt'


# 包含测试图片 (如 scene1.jpg, scene2.jpg) 的文件夹
TEST_IMAGE_DIR = 'dataset/测试集/'

# 包含真实世界坐标的 reg.json 文件
JSON_PATH = 'dataset/测试集/reg.json'

# --- 2. 加载模型和标定数据 ---

print("正在加载模型...")
try:
    # 加载你训练的标定点检测器
    dot_model = YOLO(DOT_MODEL_PATH)
    # 加载预训练的人体检测器
    # person_model = YOLO(PERSON_MODEL_PATH)
    person_model = YOLO('yolo11m.pt')
except Exception as e:
    print(f"错误: 无法加载模型。请检查路径: {e}")
    print(f"DOT_MODEL_PATH: {DOT_MODEL_PATH}")
    print(f"PERSON_MODEL_PATH: {PERSON_MODEL_PATH}")
    exit()

print("正在加载标定数据...")
try:
    with open(JSON_PATH, 'r') as f:
        # 加载JSON数据 (假设是一个列表，如 [ {"id": 1, ...}, {"id": 2, ...} ])
        calibration_data_list = json.load(f)
    
    # 将列表转换为以 'id' 为键的字典，方便快速查找
    cal_data_map = {item['id']: item for item in calibration_data_list}
    print(f"成功加载 {len(cal_data_map)} 条场景标定数据。")
    print(f"cal_data_map: {cal_data_map}")
except FileNotFoundError:
    print(f"错误: 找不到标定文件: {JSON_PATH}")
    exit()
except Exception as e:
    print(f"错误: 解析JSON文件失败: {e}")
    exit()


def get_scene_id_from_path(img_path):
    """
    辅助函数：从图像文件名 (如 '.../scene1.jpg') 中提取场景ID (如 1)
    """
    filename = os.path.basename(img_path)
    # 使用正则表达式查找文件名中的数字
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        print(f"警告: 无法从文件名 {filename} 中解析场景ID。")
        return None

def sort_points_by_y(points):
    """
    辅助函数：根据Y坐标对点列表进行排序 (从上到下)
    """
    return sorted(points, key=lambda p: p[1])
def sort_points_by_x(points):
    """
    辅助函数：根据X坐标对点列表进行排序 (从上到下)
    """
    return sorted(points, key=lambda p: p[0])

# --- 3. 遍历测试集并执行测距 ---

# 获取所有测试图片 (支持.jpg, .png, .jpeg)
test_image_paths = glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpg')) + \
                   glob.glob(os.path.join(TEST_IMAGE_DIR, '*.png')) + \
                   glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpeg'))

print(f"\n--- 开始处理 {len(test_image_paths)} 张测试图片 ---")

for img_path in test_image_paths:
    print(f"\n--- 正在处理: {os.path.basename(img_path)} ---")
    
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print("错误: 无法读取图像。")
        continue

    # 2. 获取场景ID和对应的标定数据
    scene_id = get_scene_id_from_path(img_path)
    print(f"场景id{scene_id}")
    if scene_id is None or scene_id not in cal_data_map:
        print(f"错误: 找不到ID为 {scene_id} 的标定数据，跳过此图像。")
        continue
    
    scene_data = cal_data_map[scene_id]
    
    # 3. 步骤一 (YOLO检测): 获取像素坐标
    
    # (a) 检测标定点
    dot_results = dot_model(img, verbose=False)
    src_points = []
    for box in dot_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        src_points.append([x_center.item(), y_center.item()])
        # 在图像上绘制检测到的点 (可选)
        cv2.circle(img, (int(x_center), int(y_center)), 10, (0, 255, 0), 2)

    if len(src_points) != 4:
        print(f"错误: 检测到 {len(src_points)} 个标定点 (需要4个)，跳过此图像。")
        continue

    # (b) 检测人体并获取脚部坐标
    person_results = person_model(img, classes=0, verbose=False) # classes=0: 只检测 'person'
    if len(person_results[0].boxes) == 0:
        print("错误: 未检测到人体，跳过此图像。")
        continue
    
    # 假设只取第一个检测到的人
    person_box = person_results[0].boxes[0].xyxy[0]
    px1, py1, px2, py2 = person_box
    
    # 取 BBox 底边中点作为脚部像素坐标
    foot_x_pix = (px1 + px2) / 2
    foot_y_pix = py2  # BBox 的最底边 (y坐标最大)
    
    # 格式化为 OpenCV perspectiveTransform 所需的格式 [1, N, 2]
    foot_pixel_coord = np.float32([[[foot_x_pix.item(), foot_y_pix.item()]]])

    # 在图像上绘制人体框和脚点 (可选)
    cv2.rectangle(img, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
    cv2.circle(img, (int(foot_x_pix), int(foot_y_pix)), 10, (0, 0, 255), -1)

    # 4. 步骤二 (数据对齐): 准备源坐标和目标坐标
    
    # (a) 获取真实世界坐标 (目标点)
    dst_points_list = scene_data['dot'] # 例如: [[-2.73, 5.4], [0, 2.97], ...]
    
    # (b) 关键：对齐 src 和 dst 点
    # 我们假设 reg.json 中的 'dot' 顺序 和我们检测到的 'src_points' 顺序不同。
    # 我们使用一个通用的排序方法：都按Y坐标 (从上到下) 排序，使它们一一对应。
    src_points_sorted = sort_points_by_x(src_points)
    dst_points_sorted = sort_points_by_x(dst_points_list)
    
    src_points_np = np.float32(src_points_sorted)
    dst_points_np = np.float32(dst_points_sorted)

    # 5. 步骤三 (几何变换): 计算 H 矩阵并转换坐标
    
    # (a) 计算单应性矩阵 H
    # H, _ = cv2.findHomography(src_points_np, dst_points_np)
    H = cv2.getPerspectiveTransform(src_points_np, dst_points_np)

    if H is None:
        print("错误: 无法计算单应性矩阵H。")
        continue
        
    # (b) 转换脚部坐标
    foot_world_coord = cv2.perspectiveTransform(foot_pixel_coord, H)
    
    foot_x_m = foot_world_coord[0][0][0]
    foot_y_m = foot_world_coord[0][0][1]
    
    foot_point = Point(foot_x_m, foot_y_m)
    print(f"计算得到脚部世界坐标: ({foot_x_m:.2f}, {foot_y_m:.2f}) 米")

    # 6. 步骤四 (计算距离): 使用 Shapely 计算最短距离
    
    device_areas_coords = scene_data['area'] # "area" 是一个多边形列表
    min_distance = float('inf')
    
    for area_coords in device_areas_coords:
        try:
            polygon = Polygon(area_coords)
            distance = foot_point.distance(polygon)
            if distance < min_distance:
                min_distance = distance
        except Exception as e:
            print(f"错误: 创建Shapely多边形失败: {e}")

    # 7. 打印最终结果
    print(f"**************************************************")
    print(f"*** 图像 {os.path.basename(img_path)}: 最终最短距离 = {min_distance:.4f} 米 ***")
    print(f"**************************************************")

    # 8. (可选) 可视化结果
    cv2.putText(img, f"Distance: {min_distance:.4f} m", (int(px1), int(py1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # 不弹窗，直接保存图片到文件查看
    save_dir = "/home/cfz/CV_Assignments/results"          # 目标文件夹
    name=os.path.basename(img_path)
    save_path = os.path.join(save_dir, f"{name}")    # 完整保存路径
    cv2.imwrite(save_path, img)
    cv2.waitKey(0) # 按任意键继续下一张

print("\n--- 所有图像处理完毕 ---")
# cv2.destroyAllWindows()