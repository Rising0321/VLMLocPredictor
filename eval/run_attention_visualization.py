#!/usr/bin/env python3
"""
注意力可视化使用示例脚本
展示如何使用修改后的visualizeHeatMap.py来创建注意力热力图可视化
"""

import os
import sys
from visualizeHeatMap import AttentionVisualizer

def main():
    """主函数 - 使用示例"""
    
    # 配置参数
    model_path = "checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-07-09-12-36-57/checkpoint-200"  # 你的模型路径
    benchmark_json = "trajDataJsonsDirty/sft/chengdu/pointLabel13/test-ours.json"  # 数据集路径
    image_dir = "VLM"  # 图像目录路径
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请修改model_path为正确的模型路径")
        return
    
    if not os.path.exists(benchmark_json):
        print(f"错误: 数据集文件不存在: {benchmark_json}")
        print("请修改benchmark_json为正确的数据集路径")
        return
    
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        print("请修改image_dir为正确的图像目录路径")
        return
    
    try:
        # 创建可视化器
        print("正在加载模型...")
        visualizer = AttentionVisualizer(model_path)
        
        # 加载样本数据（使用第一个样本）
        print("正在加载样本数据...")
        sample, images = visualizer.load_sample_data(benchmark_json, image_dir, sample_idx=0)
        
        print(f"样本ID: {sample['id']}")
        print(f"图像数量: {len(images)}")
        print(f"问题长度: {len(sample['problem'])} 字符")
        
        # 创建注意力可视化
        print("正在创建注意力可视化...")
        output_path = visualizer.create_attention_visualization_for_sample(
            sample=sample,
            images=images,
            layer_idx=4,  # 可视化第20层
            start_phrase="Based on the following requirements"  # 开始可视化的短语
        )
        
        print(f"\n✅ 可视化完成！")
        print(f"📁 输出文件: {output_path}")
        print(f"🎬 视频包含文本注意力和图像注意力的并排可视化")
        print(f"🔍 红色表示低注意力，绿色表示高注意力，蓝色表示当前生成的token")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

def run_multiple_samples():
    """运行多个样本的可视化示例"""
    
    model_path = "checkpoints/resume_finetune_qwen2vl_2b_task1_only_rl-2025-07-09-12-36-57/checkpoint-200"
    benchmark_json = "trajDataJsonsDirty/sft/chengdu/pointLabel13/test-ours.json"
    image_dir = "VLM"
    
    # 创建可视化器
    visualizer = AttentionVisualizer(model_path)
    
    # 为多个样本创建可视化
    sample_indices = [38-1, 109-1, 117-1, 144-1, 145-1, 183-1, 184-1]

    import os
    visiable_divice = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    
    for sample_idx in sample_indices:
        try:
            print(f"\n正在处理样本 {sample_idx}...")
            
            # 加载样本数据
            sample, images = visualizer.load_sample_data(benchmark_json, image_dir, sample_idx)
            
            # 创建注意力可视化
            output_path = visualizer.create_attention_visualization_for_sample(
                sample_idx=sample_idx,
                sample=sample,
                images=images,
                layer_idx=20,
                start_phrase="Based on the following requirements"
            )
            
            print(f"样本 {sample_idx} 可视化完成: {output_path}")
            
        except Exception as e:
            print(f"样本 {sample_idx} 处理失败: {e}")

if __name__ == "__main__":
    print("🚀 注意力可视化示例")
    print("=" * 50)
    
    # 运行单个样本示例
    run_multiple_samples()
    
    # 如果要运行多个样本，取消下面的注释
    # print("\n" + "=" * 50)
    # print("运行多个样本示例...")
    # run_multiple_samples() 