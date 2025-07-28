# VLMLocPredictor

A Vision-Language Model for Next Location Prediction in Trajectory Data.

## Overview

VLMLocPredictor is a model that combines vision-language capabilities with trajectory data to predict locations. It uses a two-stage training approach: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) based fine-tuning.

## Installation

```bash
# Clone the repository
git clone https://github.com/Rising0321/VLMLocPredictor.git
cd VLMLocPredictor

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

1. Configure your dataset paths in `data/dataset_info.json`
2. Supported datasets:
   - Chengdu: `pointLabel`, `pointLogic`
   - Porto: `pointLabelPorto`, `pointLogicPorto`
   - San Francisco: `pointLabelSanfrancisco`, `pointLogicSanfrancisco`
   - Rome: `pointLabelRome`, `pointLogicRome`

## Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

We use [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) for the SFT stage.

#### First Stage SFT
1. Set up your Vision-Language Model path as `PATH_MODEL`
2. Configure datasets:
   ```bash
   pointLabel,pointLabelPorto,pointLabelSanfrancisco,pointLabelRome
   ```
3. Run:
   ```bash
   bash scripts/train/cot_sft/resume_finetune_qwen2vl_2b_pointLabel_cot_sft.sh
   ```

#### Second Stage SFT
1. Use the model trained in First Stage as `PRETRAIN_MODEL_PATH`
2. Add logic datasets:
   ```bash
   pointLabel,pointLabelPorto,pointLabelSanfrancisco,pointLabelRome,pointLogic,pointLogicPorto,pointLogicSanfrancisco,pointLogicRome
   ```
3. Run the same script as First Stage

### Stage 2: RL-based Fine-tuning

The RL model implementation is located in `train/stage_rl/`.

1. Configure:
   - `DATASET_NAME`: Path to your dataset
   - `MODEL_NAME_OR_PATH`: Path to your pre-trained model
   - `IMAGE_PATH`: Path to your image data (will be released soon)

2. Run:
   ```bash
   bash scripts/train/reason_rft_zero/resume_finetune_qwen2vl_2b_traj_only_rl.sh
   ```

## Project Structure

```
VLMLocPredictor/
├── data/               # Dataset configuration
├── eval/               # Evaluation scripts
├── train/
│   ├── stage_sft/     # Supervised Fine-Tuning
│   └── stage_rl/      # RL-based Fine-tuning
├── scripts/           # Training scripts
└── utils/            # Utility functions
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
[Citation will be added after publication]
```

## Acknowledgements

This project builds upon several excellent open-source projects:
- [Reason-RFT](https://github.com/tanhuajie/Reason-RFT)
- [Open-R1](https://github.com/huggingface/open-r1)
