# 注意力热力图可视化

这个模块提供了用于可视化多模态模型注意力权重的工具，特别适配了Reason-RFT项目的轨迹推理任务。

## 功能特性

- 🎯 **文本注意力可视化**: 显示模型在生成文本时对不同token的注意力分布
- 🖼️ **图像注意力可视化**: 在原始图像上叠加注意力热力图，显示模型关注的图像区域
- 🎬 **视频输出**: 生成包含文本和图像注意力并排显示的视频
- 🔧 **灵活配置**: 支持选择不同的注意力层、起始短语等参数

## 文件说明

- `visualizeHeatMap.py`: 主要的可视化代码，包含所有核心功能
- `run_attention_visualization.py`: 使用示例脚本
- `README_attention_visualization.md`: 本说明文档

## 安装依赖

确保安装了以下Python包：

```bash
pip install torch transformers pillow imageio tqdm numpy
```

## 使用方法

### 方法1: 使用示例脚本

```bash
python run_attention_visualization.py
```

在运行之前，请修改脚本中的路径配置：

```python
model_path = "your/model/path"  # 你的模型路径
benchmark_json = "your/dataset.json"  # 数据集JSON文件路径
image_dir = "your/image/directory"  # 图像目录路径
```

### 方法2: 使用命令行参数

```bash
python visualizeHeatMap.py \
    --model_path "checkpoints/your_model" \
    --benchmark_json "trajDataJsonsDirty/sft/chengdu/pointLabel13/test-ours.json" \
    --image_dir "VLM" \
    --sample_idx 0 \
    --layer_idx 20 \
    --start_phrase "Based on the following requirements"
```

### 方法3: 在代码中使用

```python
from visualizeHeatMap import AttentionVisualizer

# 创建可视化器
visualizer = AttentionVisualizer("your/model/path")

# 加载样本数据
sample, images = visualizer.load_sample_data(
    "your/dataset.json", 
    "your/image/dir", 
    sample_idx=0
)

# 创建注意力可视化
output_path = visualizer.create_attention_visualization_for_sample(
    sample=sample,
    images=images,
    layer_idx=20,
    start_phrase="Based on the following requirements"
)
```

## 参数说明

### AttentionVisualizer类参数

- `model_name_or_path`: 模型路径
- `max_image_num`: 最大图像数量（默认2）

### create_attention_visualization_for_sample方法参数

- `sample`: 数据集样本字典
- `images`: PIL图像列表
- `layer_idx`: 要可视化的注意力层索引（默认20）
- `start_phrase`: 开始可视化的文本短语（默认"Based on the following requirements"）

### 命令行参数

- `--model_path`: 模型路径（必需）
- `--benchmark_json`: 基准测试JSON文件路径（必需）
- `--image_dir`: 图像目录路径（必需）
- `--sample_idx`: 要可视化的样本索引（默认0）
- `--layer_idx`: 要可视化的注意力层索引（默认20）
- `--start_phrase`: 开始可视化的短语（默认"Based on the following requirements"）

## 输出说明

### 视频内容

生成的视频包含两部分：
1. **左侧**: 文本注意力可视化
   - 红色: 低注意力权重
   - 绿色: 高注意力权重
   - 蓝色: 当前正在生成的token

2. **右侧**: 图像注意力可视化
   - 原始图像上叠加网格
   - 每个网格单元的颜色表示该区域的注意力权重
   - 红色到绿色的渐变表示注意力强度

### 文件命名

输出文件格式：`combined_attention_visualization_layer{layer_idx}.mp4`

例如：`combined_attention_visualization_layer20.mp4`

## 数据集格式

代码期望的数据集格式与Reason-RFT项目一致：

```json
[
    {
        "id": "sample_id",
        "image": ["path/to/image.png"],
        "problem": "问题文本",
        "solution": "答案文本",
        "answer": "答案文本"
    }
]
```

## 注意事项

1. **内存使用**: 生成注意力可视化需要大量内存，建议在GPU上运行
2. **模型兼容性**: 代码使用transformers库直接加载模型，确保模型支持注意力权重输出
3. **图像格式**: 支持常见的图像格式（PNG, JPG等）
4. **字体支持**: 代码会自动尝试加载等宽字体，如果失败会使用默认字体

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确保模型支持注意力权重输出

2. **内存不足**
   - 减少batch size
   - 使用更小的图像尺寸
   - 减少max_new_tokens参数

3. **字体显示问题**
   - 代码会自动尝试多种字体
   - 如果仍有问题，可以手动指定字体路径

4. **视频生成失败**
   - 确保安装了imageio和ffmpeg
   - 检查输出目录的写入权限

### 调试模式

在代码中添加调试信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在可视化器创建时添加详细日志
visualizer = AttentionVisualizer(model_path)
```

## 扩展功能

### 自定义可视化样式

可以修改`visualize_attention_step`函数来自定义文本可视化样式：

```python
# 修改颜色映射
red = int(255 * (1 - weight))
green = int(255 * weight)
blue = int(255 * weight * 0.5)  # 添加蓝色分量
color = (red, green, blue)
```

### 添加更多层可视化

可以同时可视化多个注意力层：

```python
for layer_idx in [10, 20, 30]:
    output_path = visualizer.create_attention_visualization_for_sample(
        sample, images, layer_idx, start_phrase
    )
```

## 贡献

欢迎提交问题和改进建议！ 