import json
import os
import torch
import multiprocessing
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# 固定设备为 gpu0-gpu7
num_gpus = 8
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

def load_model_and_processor(device_index):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map={"": devices[device_index]}
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    return model, processor

def check_tensor_validity(tensor, tensor_name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{tensor_name} contains invalid values (inf or nan).")
    if (tensor < 0).any():
        raise ValueError(f"{tensor_name} contains negative values.")

def preprocess_pixel_values(pixel_values):
    # 将负值裁剪为0
    pixel_values = torch.clamp(pixel_values, min=0.0)
    # 将大于1的值裁剪为1
    pixel_values = torch.clamp(pixel_values, max=1.0)
    # 如果需要，可以进行其他预处理步骤
    return pixel_values

def process_data_on_gpu(device_index, data_chunk, output_file_path):
    device = devices[device_index]
    model, processor = load_model_and_processor(device_index)
    results = []
    for data in tqdm(data_chunk, desc=f"Processing on GPU {device_index}"):
        q_text = data['q_text']
        messages = [{"role": "user", "content": [{"type": "text", "text": q_text}]}]

        # 如果需要图片输入
        if data['type'] in ["ISVGEN", "ISVGUN_usage", "ISVGUN_category", "ISVGUN_description", "ISVGUN_color"]:
            image_path = '/data_train/ryanjkli/UniSVG/dataset/svgo/png/' + data.get('image_path', None)
            messages[0]['content'].insert(0, {"type": "image", "image": image_path})
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs if image_path else None,
                videos=video_inputs if image_path else None,
                padding=True,
                return_tensors="pt",
            )
            # 检查输入张量的有效性
            check_tensor_validity(inputs.input_ids, "input_ids")
            check_tensor_validity(inputs.attention_mask, "attention_mask")

            inputs = inputs.to(device)  # 确保输入在相应的 GPU 上
            print(f"Inputs on GPU {device_index}: {inputs}")
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=8192,               
                temperature=0.7,
                top_p=0.9,
                num_beams=3,
                use_cache=True)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text[0])
            # 将生成的答案添加到数据中
            data['model_answer'] = output_text[0]
            results.append(data)
        else:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            # 检查输入张量的有效性
            check_tensor_validity(inputs.input_ids, "input_ids")
            check_tensor_validity(inputs.attention_mask, "attention_mask")

            inputs = inputs.to(device)  # 确保输入在相应的 GPU 上
            print(f"Inputs on GPU {device_index}: {inputs}")
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=8192,               
                temperature=0.7,
                top_p=0.9,
                num_beams=3,
                use_cache=True
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text[0])
            # 将生成的答案添加到数据中
            data['model_answer'] = output_text[0]
            results.append(data)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 保存结果到文件
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # 读取测评数据集
    with open('/data_train/ryanjkli/UniSVG/dataset/second_dataset/test_dataset/test.json', 'r') as f:
        test_data = json.load(f)

    output_dir = '/data_train/ryanjkli/UniSVG/dataset/second_dataset/models/finetune/origin_qwen25/results/'
    chunk_size = len(test_data) // num_gpus
    processes = []
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(test_data)
        data_chunk = test_data[start_idx:end_idx]
        output_file_path = os.path.join(output_dir, f'results_gpu_{i}.json')
        
        p = multiprocessing.Process(target=process_data_on_gpu, args=(i, data_chunk, output_file_path))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    # 汇总结果
    final_results = []
    for i in range(num_gpus):
        output_file_path = os.path.join(output_dir, f'results_gpu_{i}.json')
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as f:
                results = json.load(f)
                final_results.extend(results)
            os.remove(output_file_path)
    
    final_output_file_path = os.path.join(output_dir, 'final_results.json')
    with open(final_output_file_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Final results saved to {final_output_file_path}")

if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    multiprocessing.set_start_method('spawn')
    main()
