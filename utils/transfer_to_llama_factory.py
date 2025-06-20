import json
from tqdm import tqdm

def transform_data(input_data):
    transformed_data_list = []

    for item in tqdm(input_data):
        # 提取原始数据中的信息
        image_path = item.get("image")
        conversations = item.get("conversations")

        # 构建新的数据结构
        messages = []
        for conversation in conversations:
            message = {
                "content": conversation["value"],
                "role": "user" if conversation["from"] == "human" else "assistant"
            }
            messages.append(message)

        new_data = {
            "messages": messages
        }

        # 只有在 image_path 存在时才添加 images 键
        if image_path:
            new_data["images"] = [image_path]
        else:
            new_data["images"] = []

        transformed_data_list.append(new_data)

    return transformed_data_list

def main():
    # 原始 JSON 文件路径
    input_file_path = 'your_llava_UniSVG_json_path.json'

    # 新的 JSON 文件路径
    output_file_path = 'output_llama_UniSVG_json_path.json'

    # 读取原始 JSON 文件
    with open(input_file_path, 'r') as f:
        input_data = json.load(f)

    # 转换数据格式
    transformed_data = transform_data(input_data)

    # 保存转换后的数据到新的 JSON 文件
    with open(output_file_path, 'w') as f:
        json.dump(transformed_data, f, indent=4)

    print(f"Transformed data saved to {output_file_path}")

if __name__ == '__main__':
    main()