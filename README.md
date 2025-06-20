# UniSVG Dataset

UniSVG is a comprehensive dataset designed for unified SVG generation (from textual prompts and images) and SVG understanding (color, category, usage, etc.). It comprises 525k data items tailored for Multi-modal Large Language Models (MLLM) training and evaluation. You can access the dataset on [Hugging Face](https://huggingface.co/datasets/lili24/UniSVG).

![UniSVG Example](images/Abstract_github.png)

## 🔥 Release
### [2024/06/03]
- 🔥 **UniSVG** dataset images updated! [📂 Dataset](https://huggingface.co/datasets/lili24/UniSVG/blob/main/png.zip) [🌐 Project Page](https://ryanlijinke.github.io/) 

### [2024/05/30]
- 🔥 **UniSVG** dataset opensourced! [📂 Dataset](https://huggingface.co/datasets/lili24/UniSVG) [🌐 Project Page](https://ryanlijinke.github.io/) 

## Project Homepage

For more information, please visit the [project homepage](https://ryanlijinke.github.io/).

## Dataset Summary

Unlike bitmap images, scalable vector graphics (SVG) maintain quality when scaled, frequently employed in computer vision and artistic design in the representation of SVG code. In this era of proliferating AI-powered systems, enabling AI to understand and generate SVG has become increasingly urgent. However, AI-driven SVG understanding and generation (U&G) remain significant challenges. SVG code, equivalent to a set of curves and lines controlled by floating-point parameters, demands high precision in SVG U&G. Besides, SVG generation operates under diverse conditional constraints, including textual prompts and visual references, which requires powerful multi-modal processing for condition-to-SVG transformation. Recently, the rapid growth of Multi-modal Large Language Models (MLLMs) have demonstrated capabilities to process multi-modal inputs and generate complex vector controlling parameters, suggesting the potential to address SVG U&G tasks within a unified model. To unlock MLLM's capabilities in the SVG area, we propose an SVG-centric dataset called UniSVG, comprising 525k data items, tailored for MLLM training and evaluation. To our best knowledge, it is the first comprehensive dataset designed for unified SVG generation (from textual prompts and images) and SVG understanding (color, category, usage, etc.).
## Usage

To install the dataset, you can use the `datasets` library from Hugging Face:

```bash
pip install datasets

```
Here is an example of how to load and use the dataset:

```python
from datasets import load_dataset

# Load the dataset
UniSVG_dataset = load_dataset("lili24/UniSVG")

# Print the first example
print(UniSVG_dataset[0])
```
## Finetuning example
After downloading our UniSVG dataset, you can use your preferred models to finetune them on UniSVG/subset of UniSVG. 
We have tried to finetune on the following MLLMs, please feel free to get them: [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf), [LLaVA-LLaMA](https://huggingface.co/Intel/llava-llama-3-8b), [LLaVA-Next](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf), [GLM 4V](https://huggingface.co/THUDM/glm-4v-9b), [LLaMA 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [Qwen 2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). 
As an example, we ultized [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) frame to do the finetuning. First, please git clone your own LLaMA-Factory repo.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```
```bash
# Copy utils to your LLaMA-Factory repo.
scp your_unisvg.github.io/utils .
```
Then please transfer your downloaded UniSVG dataset into LLaMA-Factory version:
Modify and run the following two python files:
```bash
# Make sure you modify these files before using them!
python utils/transfer_to_llava.py
python utils/transfer_to_llama_factory.py
```
Then add the modified LLaMA-Factory UniSVG json into "/data", and modify the "/data/dataset_info.json" by adding:
```json
  "unisvg": {
    "file_name": "llama_UniSVG_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
```
Congrats! Your UniSVG dataset is finally ready for finetuning! We offer you an example finetuning bash file using deepspeed under LLaMA factory, please refer to: [/train/train.sh](https://github.com/Ryanlijinke/unisvg.github.io/blob/main/train/train.sh)

## Evaluation example
After finnetuning, you can edit the inference code for your model and run the inference by:
```bash
python infer.py
```

You will get a inference json file with model answers in it, then please modify and use the evaluation.py to get the final score:
```bash
python evaluation.py
```
## Acknowledgement
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), thanks for your great work!
## Citation

If you use this dataset in your research, please cite the following paper:

```bibtex
@article{li2025unisvg,
  title={UniSVG: A Unified Dataset for Vector Graphic Understanding and Generation with Multimodal Large Language Models},
  author={Li, Jinke and Yu, Jiarui and Wei, Chenxing and Dong, Hande and Lin, Qiang and Yang, Liangjing and Wang, Zhicai and Hao, Yanbin},
  year={2025}
}
