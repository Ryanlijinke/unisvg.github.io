import json

# 读取输入 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(len(data))
    return data

# 转换函数
def convert_data(input_data, id_num):
    valid_types = ['ISVGEN','ISVGUN_usage','ISVGUN_color','ISVGUN_description','ISVGUN_category']
    id_str = str(id_num).zfill(12)
    if input_data["type"] in valid_types:
        output_data = {
            "id": id_str,
            "image": get_image_path(input_data),
            "conversations": [
                {
                    "from": "human",
                    "value": input_data["q_text"]
                },
                {
                    "from": "gpt",
                    "value": input_data["a_text"]
                }
            ]
        }
    else:
        output_data = {
            "id": id_str,
            "conversations": [
                {
                    "from": "human",
                    "value": input_data["q_text"]
                },
                {
                    "from": "gpt",
                    "value": input_data["a_text"]
                }
            ]
        }
    return output_data

# 获取实际图片路径
def get_image_path(input_data):
    if input_data["image_path"]:
        return 'your_image_folder_path' + input_data["image_path"]
    else:
        return ''
# 批量转换
def convert_dataset(input_dataset):
    output_dataset = []
    for i, item in enumerate(input_dataset, start=1):
        output_dataset.append(convert_data(item, i))
    return output_dataset

# 保存到 JSON 文件
def save_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# 主函数
def main(input_file_path, output_file_path):
    # 读取输入数据集
    input_dataset = read_json_file(input_file_path)

    # 转换数据集
    output_dataset = convert_dataset(input_dataset)

    # 保存转换后的数据集
    print(len(output_dataset))
    save_json_file(output_dataset, output_file_path)

# 示例调用
input_file_path = 'your_UniSVG_json_path'  # 输入 JSON 文件路径
output_file_path = 'output_llava_UniSVG_json_path'  # 输出 JSON 文件路径

main(input_file_path, output_file_path)