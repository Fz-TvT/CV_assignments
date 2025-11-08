from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo11m.pt') 

# 开始训练
results = model.train(
    data='/home/cfz/CV_Assignments/data.yaml',  # <-- ！！修改这里！！
    epochs=400, 
    project='runs/detect',
    name='dot_detector',
    # imgsz=640,      # 分辨率翻倍
    degrees=0,        # 不旋转
    cls=0.95,
    dfl=2,
    device=-1)

print("训练完成。模型保存在:", results.save_dir)