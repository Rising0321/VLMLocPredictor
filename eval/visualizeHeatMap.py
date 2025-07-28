# 可视化attention heatmap，与图像融合
# 修改自原始代码，适配Reason-RFT项目的推理代码
import imageio.v3 as imageio
import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import argparse
import re
# register before xxxx.from_pretrained
from transformers import AutoModelForCausalLM, Qwen2VLConfig, Qwen2VLForConditionalGeneration
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)


def extract_coordinates(text):
    """从文本中提取坐标"""
    pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
    matches = re.findall(pattern, text)
    coordinates = []
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        x = x1
        y = y1
        coordinates.append((x, y))
    return coordinates

def map_back(image, height, width, x, y):
    x = x / 1000 * width
    y = y / 1000 * height
    return x, y


def visualize_attention_step(
    attention_weights_step, token_ids, processor, start_phrase="<answer>"
):
    """可视化单个步骤的注意力权重"""
    # attention_weights_step shape is [1, num_heads, seq_len, seq_len]
    # 首先去除batch维度
    attention = attention_weights_step.squeeze(0)  # 现在 [num_heads, seq_len, seq_len]

    # 获取最后一个生成token的注意力权重（最后一个位置）
    # 在注意力头之间取平均
    last_token_attention = attention[:, -1, :].mean(dim=0).detach().cpu().float()

    # 重新归一化注意力权重，使其和为1
    last_token_attention = last_token_attention / last_token_attention.sum()
    attention_weights_np = last_token_attention.numpy()

    # 应用非线性缩放以增强中等注意力权重的可见性
    # 首先归一化到[0,1]范围
    min_val = attention_weights_np.min()
    max_val = attention_weights_np.max()
    if max_val > min_val:
        normalized_weights = (attention_weights_np - min_val) / (max_val - min_val)
        # 应用幂次缩放（小于1的值将被提升）
        scaled_weights = np.power(
            normalized_weights, 0.4
        )  # 调整幂值以控制对比度
    else:
        scaled_weights = attention_weights_np

    # 首先解码所有token
    tokens = processor.batch_decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    # 通过查找起始短语找到开始索引
    full_text = "".join(tokens)
    
    # 尝试多个可能的起始短语
    possible_start_phrases = [
        "assistant"
    ]
    
    token_start_idx = 0
    found_start = False
    
    for phrase in possible_start_phrases:
        # find the  last assistant in full_text
        start_idx = full_text.rfind(phrase)
        
        if start_idx != -1:
            # 通过计算到该点的token数量将字符索引转换为token索引
            char_count = 0
            for i, token in enumerate(tokens):
                char_count += len(token)
                if char_count > start_idx:
                    token_start_idx = i
                    found_start = True
                    break
            if found_start:
                break

    # 截断token和权重，从找到的位置开始
    tokens = tokens[token_start_idx:]
    scaled_weights = scaled_weights[token_start_idx:]

    # 创建黑色图像
    img_width = 2500
    img_height = 2500
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    # 尝试加载等宽字体，如果不可用则回退到默认字体
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 50
        )
    except:
        try:
            # Windows系统字体
            font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 50)
        except:
            font = ImageFont.load_default()

    # 计算文本位置
    x, y = 40, 40
    max_width = img_width - 80
    line_height = 30

    for i, (token, weight) in enumerate(zip(tokens, scaled_weights)):
        # 处理token中的换行符 - 检查\n和{data}\n模式
        if "\n" in token or "\\n" in token:
            # 将token分割为换行符前后的部分
            parts = token.replace("\\n", "\n").split("\n")

            for j, part in enumerate(parts):
                if part:  # 如果有内容，渲染它
                    bbox = draw.textbbox((x, y), part, font=font)
                    text_width = bbox[2] - bbox[0]

                    # 检查是否需要开始新行
                    if x + text_width > max_width:
                        x = 40
                        y += line_height

                    # 对于最后一个token（当前生成），使用蓝色
                    if i == len(tokens) - 1:
                        color = (100, 150, 255)  # 当前token的浅蓝色
                    else:
                        # 基于注意力权重创建红色到绿色的渐变
                        red = int(255 * (1 - weight))
                        green = int(255 * weight)
                        color = (red, green, 0)

                    # 绘制部分
                    draw.text((x, y), part, fill=color, font=font)
                    x += text_width + 4

                # 如果有更多部分或这不是最后一个空部分，移动到下一行
                if j < len(parts) - 1 or (j == len(parts) - 1 and not part):
                    x = 40
                    y += line_height

            continue  # 跳过此token的循环其余部分

        # 正常token处理（无换行符）
        bbox = draw.textbbox((x, y), token, font=font)
        text_width = bbox[2] - bbox[0]

        # 检查是否需要开始新行
        if x + text_width > max_width:
            x = 40  # 重置x到行首
            y += line_height  # 移动到下一行

        # 对于最后一个token（当前生成），使用蓝色
        if i == len(tokens) - 1:
            color = (100, 150, 255)  # 当前token的浅蓝色
        else:
            # 基于注意力权重创建红色到绿色的渐变
            red = int(255 * (1 - weight))
            green = int(255 * weight)
            color = (red, green, 0)

        # 绘制token
        draw.text((x, y), token, fill=color, font=font)

        # 为下一个token移动x位置（在token之间添加小空间）
        x += text_width + 4

    # 将PIL图像转换为numpy数组
    return np.array(image)


def combine_attention_videos(text_frames, image_frames):
    """
    将文本注意力和图像注意力帧并排组合。

    Args:
        text_frames: 包含文本注意力可视化帧的numpy数组列表
        image_frames: 包含图像注意力可视化帧的numpy数组列表

    Returns:
        包含组合帧的numpy数组列表
    """
    assert len(text_frames) == len(image_frames), "帧数必须匹配"

    combined_frames = []
    for text_frame, image_frame in zip(text_frames, image_frames):
        # 计算目标高度（与文本帧相同）
        target_height = text_frame.shape[0]
        target_width = text_frame.shape[1] // 2  # 文本帧宽度的一半

        # 计算缩放因子以保持宽高比
        image_aspect = image_frame.shape[1] / image_frame.shape[0]
        target_aspect = target_width / target_height

        if image_aspect > target_aspect:
            # 图像比目标更宽 - 适合宽度
            new_width = target_width
            new_height = int(target_width / image_aspect)
            vertical_padding = (target_height - new_height) // 2
            horizontal_padding = 0
        else:
            # 图像比目标更高 - 适合高度
            new_height = target_height
            new_width = int(target_height * image_aspect)
            horizontal_padding = (target_width - new_width) // 2
            vertical_padding = 0

        # 调整图像大小保持宽高比
        image_frame_resized = Image.fromarray(image_frame).resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        image_frame_resized = np.array(image_frame_resized)

        # 白色背景
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)

        # 将调整大小的图像放在画布中心
        y_start = vertical_padding
        y_end = y_start + new_height
        x_start = horizontal_padding
        x_end = x_start + new_width
        canvas[y_start:y_end, x_start:x_end] = image_frame_resized

        # 水平组合帧
        combined_frame = np.hstack([text_frame, canvas])
        combined_frames.append(combined_frame)

    return combined_frames


def create_attention_visualization(
    attention_weights,
    sequences,
    processor,
    layer_idx=-1,
    fps=2,
    output_path="attention_visualization.mp4",
    start_phrase="Based on the following requirements",
):
    """
    创建生成过程中注意力权重的视频可视化。

    Args:
        attention_weights: 模型生成过程中的注意力权重列表
        sequences: 模型生成过程中的token序列
        processor: 用于解码token的tokenizer/processor
        layer_idx: 要可视化的注意力层索引（默认：-1表示最后一层）
        fps: 输出视频的帧率（默认：2）
        output_path: 保存输出视频的路径（默认："attention_visualization.mp4"）
        start_phrase: 开始可视化的短语（默认："Based on the following requirements"）
    """
    num_steps = len(attention_weights)
    base_sequence = sequences.shape[1] - num_steps

    # 在内存中存储帧
    frames = []

    for step in tqdm(range(1, num_steps)):  # 从1开始，因为步骤0只是输入
        attention_weights_step = attention_weights[step][
            layer_idx
        ]  # 获取指定层的注意力
        current_tokens = sequences[0][: base_sequence + step]
        frame = visualize_attention_step(
            attention_weights_step, current_tokens, processor, start_phrase=start_phrase
        )
        frames.append(frame)

    return frames


def visualize_image_attention(
    inputs,
    image,
    attention_weights,
    sequences,
    processor,
    generated_text="",  # 添加生成的文本参数
):
    """可视化图像注意力权重"""
    # 获取patch网格
    _, h, w = inputs["image_grid_thw"].cpu().numpy().squeeze(0)

    # 处理patch合并
    merge_size = processor.image_processor.merge_size
    h = h // merge_size
    w = w // merge_size

    total_patches = h * w

    # 输入中应该有这么多图像token
    image_pad_token = "<|image_pad|>"
    image_pad_id = processor.tokenizer.convert_tokens_to_ids(image_pad_token)

    num_image_tokens = (inputs["input_ids"] == image_pad_id).sum().cpu().numpy().item()

    assert num_image_tokens == total_patches, (
        f"Expected {num_image_tokens=} to equal {total_patches=}"
    )

    # attention_weights shape is [1, num_heads, seq_len, seq_len]
    # 首先去除batch维度
    attention = attention_weights.squeeze(0)  # 现在 [num_heads, seq_len, seq_len]

    # 获取最后一个生成token的注意力权重（最后一个位置）
    # 在注意力头之间取平均
    last_token_attention = attention[:, -1, :].mean(dim=0).detach().cpu().float()

    # 重新归一化注意力权重，使其和为1
    last_token_attention = last_token_attention / last_token_attention.sum()
    attention_weights_np = last_token_attention.numpy()

    # 现在我们应该选择对应于图像token的注意力权重
    image_tokens_mask = (inputs["input_ids"] == image_pad_id).cpu().numpy().squeeze(0)
    # 在右侧用False填充mask - 这些是生成的token
    image_tokens_mask = np.pad(
        image_tokens_mask,
        (0, attention_weights_np.shape[0] - image_tokens_mask.shape[0]),
        mode="constant",
        constant_values=False,
    )

    # 在应用mask之前，这些应该是相同的形状
    assert image_tokens_mask.shape == attention_weights_np.shape, (
        f"The image tokens mask and attention weights shape mismatch: {image_tokens_mask.shape=} {attention_weights_np.shape=}"
    )

    # 现在我们应该选择对应于图像token的注意力权重
    attention_weights_np = attention_weights_np[image_tokens_mask]

    # 每个图像token应该有一个注意力权重
    assert num_image_tokens == attention_weights_np.shape[0], (
        f"Expected {num_image_tokens=} to equal {attention_weights_np.shape[0]=}, as there should be one attention weight per image token"
    )

    # 为网格创建透明覆盖层
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 计算每个网格单元的像素大小
    width, height = image.size
    cell_width = width // w
    cell_height = height // h

    # 绘制水平线（黑色，50%透明度）
    for i in range(h + 1):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, 128), width=1)

    # 绘制垂直线（黑色，50%透明度）
    for j in range(w + 1):
        x = j * cell_width
        draw.line([(x, 0), (x, height)], fill=(0, 0, 0, 128), width=1)

    # 应用非线性缩放以增强中等注意力权重的可见性
    min_val = attention_weights_np.min()
    max_val = attention_weights_np.max()
    if max_val > min_val:
        normalized_weights = (attention_weights_np - min_val) / (max_val - min_val)
        # 应用幂次缩放（小于1的值将被提升）
        scaled_weights = np.power(normalized_weights, 0.4)
    else:
        scaled_weights = attention_weights_np

    # 用基于注意力的颜色填充每个网格单元
    for idx, weight in enumerate(scaled_weights):
        # 计算网格位置
        grid_x = idx % w
        grid_y = idx // w

        # 计算像素坐标
        x1 = grid_x * cell_width
        y1 = grid_y * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height

        # 基于注意力权重创建红色到绿色的渐变
        red = int(255 * (1 - weight))
        green = int(255 * weight)
        # 添加半透明颜色覆盖层
        draw.rectangle([x1, y1, x2, y2], fill=(red, green, 0, 128))

    # 将原始图像与覆盖层组合
    image = image.convert("RGBA")
    grid_image = Image.alpha_composite(image, overlay)

    # 提取坐标并在图像上标记
    if generated_text:
        try:
            # 提取坐标
            coordinates = extract_coordinates(generated_text)
            # if coordinates:
            #     # 在图像上标记坐标点
            #     for i, (x, y) in enumerate(coordinates):
            #         # 使用map_back函数将坐标映射到图像坐标系
            #         mapped_x, mapped_y = map_back(image, height, width, x, y)
                    
            #         # 绘制预测点（红色圆圈）
            #         circle_radius = 3
            #         circle_bbox = [
            #             mapped_x - circle_radius, 
            #             mapped_y - circle_radius,
            #             mapped_x + circle_radius, 
            #             mapped_y + circle_radius
            #         ]
                    
            #         # 绘制红色圆圈
            #         draw = ImageDraw.Draw(grid_image)
            #         draw.ellipse(circle_bbox, fill=(255, 0, 0, 200), outline=(255, 255, 255, 255), width=2)
            #         break
        except Exception as e:
            print(f"坐标提取和标记过程中发生错误: {e}")

    # 转换回RGB
    grid_image = grid_image.convert("RGB")

    grid_image = np.array(grid_image)

    return grid_image


def create_image_attention_demo(
    inputs,
    image,
    attention_weights,
    sequences,
    processor,
    generated_text="",  # 添加生成的文本参数
    layer_idx=-1,
    fps=2,
    output_path="visual_attention_demo.mp4",
):
    """
    创建图像注意力可视化演示

    Args:
        inputs: 模型的输入
        image: 传递给模型的PIL图像
        attention_weights: 模型生成过程中的注意力权重
        sequences: 模型生成过程中的生成序列
        processor: 处理器
        generated_text: 生成的完整文本，用于提取坐标
        layer_idx: 要可视化的层索引
        fps: 帧率
        output_path: 输出路径
    """
    num_steps = len(attention_weights)
    base_sequence = sequences.shape[1] - num_steps

    # 在内存中存储帧
    frames = []

    for step in tqdm(range(1, num_steps)):  # 从1开始，因为步骤0只是输入
        attention_weights_step = attention_weights[step][
            layer_idx
        ]  # 获取指定层的注意力
        current_tokens = sequences[0][: base_sequence + step]

        frame = visualize_image_attention(
            inputs, image, attention_weights_step, current_tokens, processor, generated_text
        )
        frames.append(frame)

    return frames

PROMPT = "{Question}"
PROMPT = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."



class AttentionVisualizer:
    """注意力可视化器类，适配Reason-RFT项目"""
    
    def __init__(self, model_name_or_path, max_image_num=2):
        """初始化可视化器"""
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        # 添加processor调试信息
        print(f"Processor类型: {type(self.processor)}")
        if hasattr(self.processor, 'tokenizer'):
            print(f"Tokenizer类型: {type(self.processor.tokenizer)}")
        if hasattr(self.processor, 'image_processor'):
            print(f"Image processor类型: {type(self.processor.image_processor)}")
        
        # 使用transformers直接加载模型以获取注意力权重
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.eval()
        
        self.model_name_or_path = model_name_or_path
        print(f"模型已加载: {model_name_or_path}")
    
    def load_sample_data(self, benchmark_json, image_dir, sample_idx=0):
        """加载样本数据"""
        with open(benchmark_json, 'r') as file:
            data = json.load(file)
        
        if sample_idx >= len(data):
            raise ValueError(f"样本索引 {sample_idx} 超出范围，数据集大小为 {len(data)}")
        
        sample = data[sample_idx]
        
        # 加载图像
        images = []
        if isinstance(sample["image"], list):
            for image in sample["image"]:
                path = os.path.join(image_dir, image)
                images.append(Image.open(path) if isinstance(image, str) else image)
        else:
            path = os.path.join(image_dir, sample["image"])
            images = [Image.open(path) if isinstance(sample["image"], str) else sample["image"]]
        
        return sample, images
    
    def create_attention_visualization_for_sample(self, sample, sample_idx, images, layer_idx=20, 
                                                 start_phrase="Based on the following requirements"):
        """为单个样本创建注意力可视化"""
        
        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": PROMPT.format(Question=sample['problem'])},
                ],
            }
        ]

        # 应用chat模板获取文本
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 正确处理图像输入
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # 添加调试信息
        print(f"输入张量信息:")
        print(f"  input_ids shape: {inputs['input_ids'].shape}")
        print(f"  attention_mask shape: {inputs['attention_mask'].shape}")
        if 'image_grid_thw' in inputs:
            print(f"  image_grid_thw shape: {inputs['image_grid_thw'].shape}")
        if 'image_patches' in inputs:
            print(f"  image_patches shape: {inputs['image_patches'].shape}")

        print("开始生成...")
        try:
            with torch.no_grad():
                generated_output = self.model.generate(
                    **inputs,
                    temperature=0.1,
                    max_new_tokens=4096,  # 减少token数量以避免内存问题
                    output_attentions=True,
                    return_dict_in_generate=True,
                    use_cache=True,  # 启用缓存
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            print("生成完成")
        except RuntimeError as e:
            print(f"生成过程中发生错误: {e}")
            print("尝试使用更保守的设置...")
            try:
                with torch.no_grad():
                    generated_output = self.model.generate(
                        **inputs,
                        temperature=0.1,
                        max_new_tokens=4096,  # 进一步减少token数量
                        output_attentions=True,
                        return_dict_in_generate=True,
                        use_cache=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        do_sample=False,  # 使用贪婪解码
                    )
                print("使用保守设置生成完成")
            except RuntimeError as e2:
                print(f"仍然失败: {e2}")
                # 尝试不使用attention输出
                print("尝试不使用attention输出...")
                try:
                    with torch.no_grad():
                        generated_output = self.model.generate(
                            **inputs,
                            temperature=0.1,
                            max_new_tokens=4096,
                            output_attentions=False,  # 不输出attention
                            return_dict_in_generate=True,
                            use_cache=True,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            do_sample=False,
                        )
                    print("生成完成（无attention输出）")
                    print("注意：由于无法获取attention权重，将跳过可视化")
                    return None
                except RuntimeError as e3:
                    print(f"所有尝试都失败: {e3}")
                    raise e3

        # 打印生成的文本
        generated_text = self.processor.decode(
            generated_output.sequences[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print("生成的文本:")
        print(generated_text)

        # 检查是否有attention权重
        if not hasattr(generated_output, 'attentions') or generated_output.attentions is None:
            print("警告：没有获取到attention权重，无法创建可视化")
            return None

        # 为指定层创建可视化
        print(f"为第 {layer_idx} 层创建可视化...")
        
        # 检查层索引是否有效
        if layer_idx >= len(generated_output.attentions[0]):
            print(f"警告：层索引 {layer_idx} 超出范围，最大层数为 {len(generated_output.attentions[0])}")
            layer_idx = len(generated_output.attentions[0]) - 1
            print(f"使用最后一层: {layer_idx}")
       
        # 获取两种可视化的帧
        text_frames = create_attention_visualization(
            generated_output.attentions,
            generated_output.sequences,
            self.processor,
            layer_idx=layer_idx,
            start_phrase="assistant",  # 从assistant开始显示
        )

        image_frames = create_image_attention_demo(
            inputs,
            images[0],
            generated_output.attentions,
            generated_output.sequences,
            self.processor,
            generated_text=generated_text,  # 传递生成的文本
            layer_idx=layer_idx,
        )

        # 组合帧并保存视频
        combined_frames = combine_attention_videos(text_frames, image_frames)
        
        # 确保videos目录存在
        os.makedirs("videos", exist_ok=True)
        
        output_path = f"videos/{sample_idx}-combined_attention_visualization_layer{layer_idx}.mp4"
        imageio.imwrite(output_path, combined_frames, fps=2, codec="libx264")
        
        print(f"可视化已保存到: {output_path}")
        
        return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="创建注意力热力图可视化")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--benchmark_json", type=str, required=True, help="基准测试JSON文件路径")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录路径")
    parser.add_argument("--sample_idx", type=int, default=0, help="要可视化的样本索引")
    parser.add_argument("--layer_idx", type=int, default=20, help="要可视化的注意力层索引")
    parser.add_argument("--start_phrase", type=str, default="Based on the following requirements", 
                       help="开始可视化的短语")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = AttentionVisualizer(args.model_path)
    
    # 加载样本数据
    sample, images = visualizer.load_sample_data(args.benchmark_json, args.image_dir, args.sample_idx)
    
    # 创建注意力可视化
    output_path = visualizer.create_attention_visualization_for_sample(
        sample=sample,
        sample_idx=args.sample_idx,
        images=images,
        layer_idx=args.layer_idx,
        start_phrase=args.start_phrase
    )
    
    print(f"可视化完成！输出文件: {output_path}")


if __name__ == "__main__":
    main()