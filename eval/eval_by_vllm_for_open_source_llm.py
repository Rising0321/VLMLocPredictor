import json, os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import re

# ====================================
#  COT PROMPT
# ====================================

def extract_coordinates(text):
    # Extract coordinates from text using regex
    pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
    matches = re.findall(pattern, text)
    coordinates = []
    for match in matches:
        # Convert to integers and take average of the box coordinates
        x1, y1, x2, y2 = map(int, match)
        x = x1
        y = y1
        coordinates.append((x, y))
    return coordinates

COT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provxided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.
   
### Output Format  

You should first thinks about the reasoning process internally and then provides the user with the answer. The **reasoning process** and **answer** are enclosed within specific tags:  

- **Reasoning process**: Enclosed within `<think>...</think>`  
- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`  

Now, it's your turn!

{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

COT_CLEVR_MATH_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

COT_GEOMATH_QUESTION_PROMPT = "{Question}  Output the thinking process in <think> </think> and final answer (number or choice) in <answer> </answer> tags."

COT_GEOMETRY_QUESTION_PROMPT = "{Question} Output the thinking process in <think> </think> and final answer (number or choice) in <answer> </answer> tags."

# todo: 根据任务加上output
COT_TrajClassification_QUESTION_PROMPT = "{Question}"
# COT_TrajClassification_QUESTION_PROMPT = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."


COT_TRANCE_QUESTION_WITH_CAPTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.
   
### Output Format  

You should first thinks about the reasoning process internally and then provides the user with the answer. The **reasoning process** and **answer** are enclosed within specific tags:  

- **Summary process**: Summary how you will approach the problem and explain the steps you will take to reach the answer, enclosed within `<summary>...</summary>`

- **Caption process**: Provide a detailed description of the image, particularly emphasizing the aspects related to the question, enclosed within `<caption>...</caption>`

- **Reasoning process**: Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning, enclosed within `<think>...</think>`  

- **Final answer (sequence of functions only)**: Enclosed within `<answer>...</answer>`

Now, it's your turn!

{Question} Output the summary process in <summary> </summary>, caption process in <caption>...</caption>, thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

# ====================================
#  SFT PROMPT
# ====================================

SFT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.

Now, it's your turn!

{Question}
'''

SFT_CLEVR_MATH_QUESTION_PROMPT = "{Question}"

SFT_GEOMATH_QUESTION_PROMPT = "{Question}"

SFT_GEOMETRY_QUESTION_PROMPT = "{Question}"

# ====================================
#  Zero-Shot PROMPT
# ====================================

ZERO_SHOT_TRANCE_QUESTION_PROMPT = '''Your need to complete the spatial visual reasoning task according to the following rules.  

Given the image of the initial state, the image of the final state, and the attributes of the initial objects, you should determine a transformation that can achieve the change of states.  

The **attributes of the initial objects** are provided as a list of tuples in the following format:  
**('object_id', 'shape', 'size', 'color', 'material')**  
Each tuple represents an object and its properties in the initial state.  

The transformation should be a sequence of functions with a length ranging from 1 to 4, where each function is represented as **'func(object_id, value)'**.  

### Available functions and values:  

1. **'change_size(object_id, value)'** - Changes the object to a new size relative to its initial size.  
   - Possible values: `['small', 'medium', 'large']`  

2. **'change_color(object_id, value)'** - Changes the object to a new color relative to its initial color.  
   - Possible values: `['yellow', 'gray', 'cyan', 'blue', 'brown', 'green', 'red', 'purple']`  

3. **'change_material(object_id, value)'** - Changes the object to a new material relative to its initial material.  
   - Possible values: `['glass', 'metal', 'rubber']`  

4. **'change_shape(object_id, value)'** - - Changes the object to a new shape relative to its initial shape.  
   - Possible values: `['cube', 'sphere', 'cylinder']`  

5. **'change_position(object_id, value)'** - Moves the object to a new position relative to its initial location.  
   - Possible values: `['front', 'behind', 'left', 'right', 'front_left', 'front_right', 'behind_left', 'behind_right']`  
   - 'front' means moving forward along the object's initial direction.  
   - 'behind' means moving backward along the object's initial direction.  
   - 'left' means moving to the left of the object's initial orientation.  
   - 'right' means moving to the right of the object's initial orientation.  
   - 'front_left' means moving diagonally toward the front and left of the initial location.  
   - 'front_right' means moving diagonally toward the front and right of the initial location.  
   - 'behind_left' means moving diagonally toward the behind and left of the initial location.  
   - 'behind_right' means moving diagonally toward the behind and right of the initial location.

Now, it's your turn!

{Question} Please output the answer only with a sequence of functions for transformation.
'''

ZERO_SHOT_CLEVR_MATH_QUESTION_PROMPT = "Please answer in Arabic numerals. For example, if the answer is 3, please respond with 3. {Question}"

ZERO_SHOT_GEOMATH_QUESTION_PROMPT = "Please answer the question with only numbers (either integer or float, such as 1, 2, 5.2, etc.) or options (such as A, B, C, or D). If it is an option, please provide your answer as a single letter (A, B, C, or D). For example, if the answer is A, just respond with A. Do not include any explanations or additional text. {Question}"

class LLM_Evaluator():
    def __init__(self, model_name_or_path):
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=768,
        )

        self.model_name_or_path = model_name_or_path

    def eval_batch(self, sample_list):
        prompts_text = []
        for sample in sample_list:
            # texts
            if self.task_name == "geomath" and self.eval_type in ["sft", "zero-shot"]:
                prompt_text = self.prompt.format(Question=sample['problem_no_prompt'])
            else:
                prompt_text = self.prompt.format(Question=sample['problem'])

            prompts_text.append(prompt_text)

        outputs = self.model.generate(prompts_text, sampling_params=self.sampling_params, use_tqdm=False)

        assert len(outputs) == len(prompts_text), f"Out({len(outputs)}) != In({len(prompts_text)})"

        for output, item in zip(outputs, sample_list):
            generated_text = output.outputs[0].text
            item["pred"] = generated_text

        return sample_list
    
    def run(self, task_name="trance", eval_type="cot-sft", batch=16):

        assert task_name in ["trance", "trance-left", "trance-right", "clevr-math",
                             "TrajClassification", "pointLabel", "pointLabel13", "pointLabelLLM", "Trajectory",
                             "super-clevr", "geomath", "geometry3k"], \
            f"Task ({task_name}) is not supported. Please choose in ['trance', 'trance-left', 'trance-right', 'clevr-math', 'super-clevr', 'geomath', 'geometry3k']"
        
        assert eval_type in ["zero-shot", "sft", "cot-sft", "caption-cot"], f"Type ({eval_type}) is not supported. Please choose in ['zero-shot', 'sft', 'cot-sft']"

        # Prompt
        self.prompt = COT_TrajClassification_QUESTION_PROMPT

        # Path to benchmark
        if task_name == "trance":
            self.benchmark_json = "/path/to/your/benchmarks/spatial_transformation/trance.json"
        elif task_name == "trance-left":
            self.benchmark_json = "/path/to/your/benchmarks/trance/trance_left.json"
        elif task_name == "trance-right":
            self.benchmark_json = "/path/to/your/benchmarks/trance/trance_right.json"
        elif task_name == "geomath":
            self.benchmark_json = "/path/to/your/benchmarks/structure_perception/geomath.json"
        elif task_name == "geometry3k":
            self.benchmark_json = "/path/to/your/benchmarks/structure_perception/geometry3k.json"
        elif task_name == "clevr-math":
            self.benchmark_json = "/root/private_data/Reason-RFT/Reason-RFT-CoT-Dataset/test_jsons/Visual-Counting-id-test-1k.json"
        elif task_name == "super-clevr":
            self.benchmark_json = "/path/to/your/benchmarks/visual_counting/super_clevr.json"
        elif task_name == "TrajClassification":
            self.benchmark_json = "/root/private_data/Reason-RFT/trajDataJsonsDirty/sft/chengdu/trajClassification/test-ours.json"
        elif task_name == "pointLabel":
            self.benchmark_json = f"/root/private_data/Reason-RFT/trajDataJsonsDirty/sft/{args.city}/pointLabelRandom/test-ours.json"
        elif task_name == "pointLabel13":
            self.benchmark_json = f"/root/private_data/Reason-RFT/trajDataJsonsDirty/sft/{args.city}/pointLabel13/test-ours.json"
        elif task_name == "pointLabelLLM":
            self.benchmark_json = f"/root/private_data/Reason-RFT/trajDataJsonsDirty/sft/{args.city}/pointLabelLLM/test-ours.json"

        self.task_name = task_name
        self.eval_type = eval_type

        with open(self.benchmark_json, 'r') as file:
            data = json.load(file)

        sample_batch = []
        data_with_pred = []
        pred_times = 0

        for idx, sample in tqdm(enumerate(data), desc=f"{self.task_name}-{self.eval_type}", total=len(data)):
            sample_batch.append(sample)

            if idx % batch != batch - 1 and idx != len(data) - 1:
                continue
            else:
                sample_batch_with_pred = self.eval_batch(sample_batch)
                data_with_pred += sample_batch_with_pred
                pred_times += 1
                sample_batch = []

            if pred_times % 10 == 0:
                self.path_to_save = os.path.join(self.model_name_or_path, f"{args.city}-llm-result")
                if not os.path.exists(self.path_to_save):
                    os.makedirs(self.path_to_save)
                
                with open(os.path.join(self.path_to_save, f"{self.task_name}.json"), 'w', encoding='utf-8') as outfile:
                    json.dump(data_with_pred, outfile, indent=4)

        with open(os.path.join(self.path_to_save, f"{self.task_name}.json"), 'w', encoding='utf-8') as outfile:
            json.dump(data_with_pred, outfile, indent=4)

        print(f"Save to {os.path.join(self.path_to_save, f'{self.task_name}.json')}")


class QWEN_LLM_Evaluator(LLM_Evaluator):
    def __init__(self, model_name_or_path):
        self.model = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
        )

        self.model_name_or_path = model_name_or_path


if __name__ == "__main__":

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Evaluate a language model on different benchmarks with specified strategies.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation (default: 16)")
    parser.add_argument('--city', type=str, default="chengdu", help="which city to eval")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--benchmark_list', type=str, nargs='+', default=["trance", "clevr-math", "super-clevr", "geomath"], help="List of benchmarks to evaluate on.")
    parser.add_argument('--stratage_list', type=str, nargs='+', default=["cot-sft", "cot-sft", "cot-sft", "cot-sft"], help="List of strategies for each benchmark.")

    # Parse the arguments
    args = parser.parse_args()

    print(f"Benchmark List: {args.benchmark_list}")
    print(f"Stratage List: {args.stratage_list}")

    print(f"Loading Model Path from {args.model_name_or_path} ...")
    if 'qwen' in args.model_name_or_path.lower():
        print("======== Using QWEN_LLM_Evaluator ==========")
        evaluator = QWEN_LLM_Evaluator(args.model_name_or_path)
    else:
        print("======== Using Default LLM_Evaluator ==========")
        evaluator = LLM_Evaluator(args.model_name_or_path)

    for benchmark, stratage in zip(args.benchmark_list, args.stratage_list):
        print(f"================== Evaluating {benchmark}-{stratage} ==================")
        evaluator.run(benchmark, stratage, args.batch_size) 