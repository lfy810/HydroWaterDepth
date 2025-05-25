水尺图像识别与水深计算系统
该项目基于 YOLOv8 实现图像中水尺的识别和水深推算，支持对静态水尺图片进行自动检测和测深，可应用于智能水文监测系统。

项目功能
自动识别图像中的水面线（基于 waterline.pt 模型）

检测水尺上的主数字（0-9）及其垂直位置（基于 best.pt 模型）

结合水面线位置，进行水深值的插值或外推计算

可视化水尺、检测结果及最终水深

最大误差控制在 ±1 cm 内

技术栈
Python 3.x

Ultralytics YOLOv8

OpenCV

Matplotlib

NumPy

项目结构
├── main.py                 # 主程序，包含图像读取、模型预测、插值计算、可视化
├── waterline.pt            # 训练好的水面线识别模型
├── train8/weights/best.pt  # 训练好的数字+E刻度识别模型
└── demoImage/              # 示例图片文件夹

将待处理的水尺图片路径填写在 main.py 的 IMAGE_PATH 变量中

运行主程序

终端将输出如下格式的推算结果：
main_digits(cm, y): [(60, 523.0), (70, 615.0)]
⛲ 推算水深约：63.4 cm
