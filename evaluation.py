import json
import re
import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
from transformers import CLIPProcessor, CLIPModel
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import cairosvg
import signal

input_folder = 'Your_test_image_folder'
igen_output_folder = 'Your_ISVGEN_results_folder'
tgen_output_folder = 'Your_TSVGEN_results_folder'
test_dataset_path = 'Your_UniSVG_test_json_path'
model_answer_path = 'Your_Model_answer_json_path'
output_json_path = 'Your_final_score_output_path'

# 初始化 LPIPS 模型
lpips_model = lpips.LPIPS(net='alex')

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_path = "openai/clip-vit-large-patch14-336"
clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

# 加载 Sentence-BERT 模型
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# 定义评价函数
def evaluate_images(original_image_path, generated_image_path):
    original_image = Image.open(original_image_path).convert('RGB')
    generated_image = Image.open(generated_image_path).convert('RGB')

    original_image_np = np.array(original_image)
    generated_image_np = np.array(generated_image)

    # 确定 win_size
    min_dim = min(original_image_np.shape[0], original_image_np.shape[1], generated_image_np.shape[0], generated_image_np.shape[1])
    win_size = min(7, min_dim // 2 * 2 + 1)  # 确保 win_size 是奇数且不超过最小边长

    # 计算 SSIM
    ssim_value = ssim(original_image_np, generated_image_np, multichannel=True, win_size=win_size, channel_axis=2)

    # 计算 PSNR
    mse_value = np.mean((original_image_np - generated_image_np) ** 2)
    if mse_value == 0:
        psnr_value = float('inf')
    else:
        psnr_value = 10 * np.log10((255 ** 2) / mse_value)

    # 计算 LPIPS
    original_image_tensor = lpips.im2tensor(original_image_np)
    generated_image_tensor = lpips.im2tensor(generated_image_np)
    lpips_value = lpips_model(original_image_tensor, generated_image_tensor).item()

    # 计算 CLIP 相似性
    inputs = clip_processor(text=["a photo"], images=[original_image, generated_image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        clip_similarity = torch.cosine_similarity(image_features[0], image_features[1], dim=0).item()

    return ssim_value, psnr_value, lpips_value, clip_similarity

# 读取测试集数据
with open(test_dataset_path, 'r') as f:
    dataset = json.load(f)

# 将测试集数据转换为字典，便于查找
dataset_dict = {(item['q_text'], item['image_path']): item for item in dataset}

# 读取模型回答数据
with open(model_answer_path, 'r') as f:
    gpt_output = json.load(f)

# 初始化计数器
type_metrics = {}
type_samples = {}
missing_samples = []
generated_samples_count = 0

# 定义一个函数来提取答案中的数字
def extract_numbers(text):
    return [float(num) for num in re.findall(r'\d+\.\d+|\d+', text)]

# 定义一个函数来提取答案中的首字母大写单词
def extract_capitalized_words(text):
    return re.findall(r'\b[A-Z][a-z]*\b', text)

# 定义保存 SVG 文件和转换为 PNG 文件的函数
def save_svg_and_convert_to_png(svg_content, output_path):
    svg_path = output_path + '.svg'
    png_path = output_path + '.png'
    
    # 保存 SVG 文件
    with open(svg_path, 'w') as f:
        f.write(svg_content)
    
    # 转换 SVG 文件为 PNG 文件
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=336, output_height=336, background_color='white')
    except Exception as e:
        print(f"Error converting {svg_path} to PNG: {e}")
        return None
    
    return png_path

# 定义修复不完整 SVG 标签的函数
def fix_svg(svg_content):
    # 删除最后一个未完成的标签
    svg_content = re.sub(r'<[^>]*$', '', svg_content)
    
    # 找到所有未闭合的标签
    open_tags = re.findall(r'<([a-zA-Z]+)[^>]*>', svg_content)
    close_tags = re.findall(r'</([a-zA-Z]+)>', svg_content)
    
    # 补齐未闭合的标签
    open_tag_stack = []
    for tag in open_tags:
        if tag not in close_tags:
            open_tag_stack.append(tag)
        else:
            close_tags.remove(tag)
    
    while open_tag_stack:
        tag = open_tag_stack.pop()
        svg_content += f'</{tag}>'
    
    # 补齐 <svg> 标签
    if not svg_content.strip().endswith('</svg>'):
        svg_content += '</svg>'
    
    return svg_content

# 定义保存中间结果的函数
def save_intermediate_results(output_json_path, final_results):
    with open(output_json_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Intermediate results saved to {output_json_path}")

# 定义解析基本形状的函数
basic_shapes = ["rect", "circle", "ellipse", "line", "polygon", "polyline", "path", "text"]

def parse_shapes(text):
    """
    解析文本中的基本形状及其数量
    """
    shape_counts = {shape: 0 for shape in basic_shapes}
    pattern = re.compile(r"(\d+)\s+(" + "|".join(basic_shapes) + r")s?")
    matches = pattern.findall(text)
    for count, shape in matches:
        shape_counts[shape] = int(count)
    return shape_counts

def calculate_accuracy(a_text, model_answer):
    """
    计算模型回答的准确性
    """
    a_shapes = parse_shapes(a_text)
    model_shapes = parse_shapes(model_answer)
    
    correct_shapes = 0
    total_shapes = sum(1 for shape in basic_shapes if a_shapes[shape] > 0)
    
    for shape in basic_shapes:
        if a_shapes[shape] > 0:
            if a_shapes[shape] == model_shapes[shape]:
                correct_shapes += 1
    
    if total_shapes == 0:
        return 1.0 if sum(model_shapes.values()) == 0 else 0.0
    return correct_shapes / total_shapes

# 定义提取宽度和高度的函数
def extract_dimensions(text):
    """
    提取文本中的宽度和高度，包括小数部分
    """
    numbers = extract_numbers(text)
    if len(numbers) >= 2:
        return numbers[:2]
    return None, None

# 定义提取变换和旋转的函数
def extract_transformations(text):
    """
    提取文本中的变换和旋转的数量
    """
    transformations = ["translate", "rotate", "scale", "skewX", "skewY", "matrix"]
    transform_counts = {transform: 0 for transform in transformations}
    for transform in transformations:
        transform_counts[transform] = len(re.findall(transform, text))
    return transform_counts

# 将模型回答数据转换为字典，便于查找
gpt_output_dict = {(item['q_text'], item['image_path']): item for item in gpt_output}

# 定义超时处理函数
def handler(signum, frame):
    raise TimeoutError("Processing timed out")

# 设置超时处理函数
signal.signal(signal.SIGALRM, handler)

# 遍历测试集数据
total_samples = len(dataset)
for idx, test_sample in enumerate(dataset):
    q_text = test_sample['q_text']
    image_path = test_sample['image_path']
    type = test_sample['type']
    correct_answer = test_sample['a_text']
    
    if type not in type_metrics:
        type_metrics[type] = {
            'ssim': 0,
            'psnr': 0,
            'lpips': 0,
            'clip_score': 0,
            'bertscore': 0,
            'sbert_score': 0,
            'accuracy': 0
        }
        type_samples[type] = 0
    
    model_output = gpt_output_dict.get((q_text, image_path), None)
    
    if model_output is None:
        # 如果找不到模型输出，记录该样本并设置评价指标为最大或最小值
        missing_samples.append(test_sample)
        if type in {"CSVGUN_color","CSVGUN_category","ISVGUN_color","ISVGUN_category"}:
            type_metrics[type]['accuracy'] += 0
        elif type in {"ISVGEN", "TSVGEN"}:
            type_metrics[type]['ssim'] += 0
            type_metrics[type]['psnr'] += 0
            type_metrics[type]['lpips'] += 1
            type_metrics[type]['clip_score'] += 0
        elif type in {"CSVGUN_usage","ISVGUN_usage","CSVGUN_rect","CSVGUN_circle","CSVGUN_description","ISVGUN_description"}:
            type_metrics[type]['bertscore'] += 0
            type_metrics[type]['sbert_score'] += 0
        elif type == "CSVGUN_shape":
            type_metrics[type]['accuracy'] += 0
        elif type == "CSVGUN_size":
            type_metrics[type]['accuracy'] += 0
        elif type == "CSVGUN_transform":
            type_metrics[type]['accuracy'] += 0
        type_samples[type] += 1
        continue
    
    model_answer = model_output['model_answer']
    generated_samples_count += 1
    
    try:
        # 设置超时时间为60秒
        signal.alarm(60)
        
        if type in {"CSVGUN_color","ISVGUN_color"}:
            # 提取模型答案中的颜色，并去重
            model_colors = set(re.findall(r'\b\w+\b', model_answer.lower()))
            correct_colors = set(color.lower() for color in test_sample['keywords']['colors']['colors'] if color.lower() != 'white')
            
            # 计算匹配的颜色数量
            matched_colors = sum(1 for color in model_colors if color in correct_colors)
            if len(correct_colors) == 0:
                print(f'颜色不对: {image_path}')
            type_metrics[type]['accuracy'] += matched_colors / len(correct_colors)
            type_samples[type] += 1
        
        elif type in {"CSVGUN_category","ISVGUN_category"}:
            # 将标准答案变成小写和首字母大写
            correct_categories_lower = set(category.lower() for category in test_sample['keywords']['category'])
            correct_categories_capitalized = set(category.capitalize() for category in test_sample['keywords']['category'])
            
            # 计算匹配的类别数量
            matched = any(category in model_answer for category in correct_categories_lower) or \
                      any(category in model_answer for category in correct_categories_capitalized)
            type_metrics[type]['accuracy'] += 1 if matched else 0
            type_samples[type] += 1
        
        elif type in {"ISVGEN", "TSVGEN"}:
            # 读取图像路径
            image_path = os.path.join(input_folder, test_sample['image_path'])
            
            # 确保模型答案是字符串
            if isinstance(model_answer, str):
                # 提取 <svg> 标签内的内容
                svg_content_match = re.search(r'<svg[^>]*>(.*?)</svg>', model_answer, re.DOTALL)
                if svg_content_match:
                    svg_content = svg_content_match.group(0)
                else:
                    print(f"No valid <svg> content found in model answer for {test_sample['image_path']}. Attempting to fix.")
                    # 尝试修复不完整的 SVG 标签
                    svg_content = re.search(r'<svg[^>]*>.*', model_answer, re.DOTALL)
                    if svg_content:
                        svg_content = fix_svg(svg_content.group(0))
                    else:
                        svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="336" height="336"></svg>'
                
                # 确定输出文件夹
                if type == "ISVGEN":
                    output_folder = igen_output_folder
                elif type == "TSVGEN":
                    output_folder = tgen_output_folder
                
                # 保存模型回答为 SVG 文件并转换为 PNG 文件
                output_path = os.path.join(output_folder, os.path.splitext(test_sample['image_path'])[0])
                os.makedirs(output_folder, exist_ok=True)
                generated_image_path = save_svg_and_convert_to_png(svg_content, output_path)
                
                if generated_image_path is None:
                    continue
                
                # 打印调试信息
                print(f"Evaluating sample {idx + 1}/{total_samples}")
                print(f"Original image path: {image_path}")
                print(f"Generated image path: {generated_image_path}")
                
                # 计算图像相似度和其他指标
                try:
                    ssim_value, psnr_value, lpips_value, clip_similarity = evaluate_images(image_path, generated_image_path)
                except ValueError as e:
                    print(f"Error evaluating {image_path} and {generated_image_path}: {e}")
                    ssim_value, psnr_value, lpips_value, clip_similarity = 0, 0, 1, 0  # 设置为0或最低值
                
                type_metrics[type]['ssim'] += ssim_value
                type_metrics[type]['psnr'] += psnr_value if psnr_value != float('inf') else 0
                type_metrics[type]['lpips'] += lpips_value
                type_metrics[type]['clip_score'] += clip_similarity
                type_samples[type] += 1
        
        elif type in {"CSVGUN_usage","ISVGUN_usage","CSVGUN_rect","CSVGUN_circle","CSVGUN_description","ISVGUN_description"}:
            # 确保模型答案和标准答案不是 None 或空字符串
            if model_answer and test_sample['a_text']:
                # 计算 BERTScore
                P, R, F1 = bert_score([model_answer], [test_sample['a_text']], lang="en", rescale_with_baseline=True)
                bertscore = F1.mean().item()
                
                # 计算 Sentence-BERT 相似度
                model_embedding = sbert_model.encode(model_answer, convert_to_tensor=True)
                correct_embedding = sbert_model.encode(test_sample['a_text'], convert_to_tensor=True)
                sbert_score = util.pytorch_cos_sim(model_embedding, correct_embedding).item()
                
                type_metrics[type]['bertscore'] += bertscore
                type_metrics[type]['sbert_score'] += sbert_score
            else:
                type_metrics[type]['bertscore'] += 0
                type_metrics[type]['sbert_score'] += 0
            type_samples[type] += 1
        
        elif type == "CSVGUN_shape":
            # 计算模型回答的准确性
            accuracy = calculate_accuracy(correct_answer, model_answer)
            type_metrics[type]['accuracy'] += accuracy
            type_samples[type] += 1

        elif type == "CSVGUN_size":
            # 提取模型答案和标准答案中的宽度和高度
            model_width, model_height = extract_dimensions(model_answer)
            correct_width, correct_height = extract_dimensions(correct_answer)
            
            # 计算宽度和高度的匹配情况
            accuracy = 0
            if model_width == correct_width and model_height == correct_height:
                accuracy = 1
            elif model_width == correct_width or model_height == correct_height:
                accuracy = 0.5
            
            type_metrics[type]['accuracy'] += accuracy
            type_samples[type] += 1

        elif type == "CSVGUN_transform":
            # 提取模型答案和标准答案中的变换和旋转的数量
            model_transforms = extract_transformations(model_answer)
            correct_transforms = extract_transformations(correct_answer)
            # print(model_transforms)
            # print(correct_transforms)
            # 计算每种变换的匹配情况
            correct_count = 0
            total_count = sum(correct_transforms.values())
            # print(total_count)
            for transform in correct_transforms:
                if transform in model_transforms:
                    correct_count += min(model_transforms[transform], correct_transforms[transform])
            
            # 计算准确性
            if total_count > 0:
                accuracy = correct_count / total_count
            else:
                accuracy = 1.0 if correct_count == 0 else 0.0
            # print(accuracy)
            type_metrics[type]['accuracy'] += accuracy
            type_samples[type] += 1

    except TimeoutError:
        print(f"Processing sample {idx + 1}/{total_samples} timed out")
        missing_samples.append(test_sample)
        continue
    except Exception as e:
        print(f"Error processing sample {idx + 1}/{total_samples}: {e}")
        missing_samples.append(test_sample)
        continue

    finally:
        # 取消超时警报
        signal.alarm(0)

    # 打印当前进度和累积指标
    print(f"Processed {idx + 1}/{total_samples} samples ({(idx + 1) / total_samples * 100:.2f}%)")
    for t in type_metrics:
        if type_samples[t] > 0:
            print(f"Type: {t}")
            if t in {"CSVGUN_color","CSVGUN_category","ISVGUN_color","ISVGUN_category","CSVGUN_size", "CSVGUN_shape","CSVGUN_transform"}:
                # 输出颜色相关指标
                average_accuracy = type_metrics[t]['accuracy'] / type_samples[t]
                print(f"  accuracy: {average_accuracy:.2f}")
            elif t in {"ISVGEN", "TSVGEN"}:
                # 输出图像相关指标
                average_ssim = type_metrics[t]['ssim'] / type_samples[t]
                average_psnr = type_metrics[t]['psnr'] / type_samples[t]
                average_lpips = type_metrics[t]['lpips'] / type_samples[t]
                average_clip_score = type_metrics[t]['clip_score'] / type_samples[t]
                print(f"  ssim: {average_ssim:.2f}")
                print(f"  psnr: {average_psnr:.2f}")
                print(f"  lpips: {average_lpips:.2f}")
                print(f"  clip_score: {average_clip_score:.2f}")
            elif t in {"CSVGUN_usage","ISVGUN_usage","CSVGUN_rect","CSVGUN_circle","CSVGUN_description","ISVGUN_description"}:
                # 输出文本相关指标
                average_bertscore = type_metrics[t]['bertscore'] / type_samples[t]
                average_sbert_score = type_metrics[t]['sbert_score'] / type_samples[t]
                print(f"  bertscore: {average_bertscore:.2f}")
                print(f"  sbert_score: {average_sbert_score:.2f}")
        else:
            print(f"Type: {t}, No samples found")

# 最终计算并打印每种类型的平均分数
final_results = {}
final_results['total_samples'] = total_samples
final_results['generated_samples_count'] = generated_samples_count
final_results['type_samples'] = type_samples
final_results['type_metrics'] = {}
final_results['missing_samples'] = missing_samples

print("\nFinal Results:")
for t in type_metrics:
    if type_samples[t] > 0:
        print(f"Type: {t}")
        final_results['type_metrics'][t] = {}
        if t in {"CSVGUN_color","CSVGUN_category","ISVGUN_color","ISVGUN_category","CSVGUN_size", "CSVGUN_shape","CSVGUN_transform"}:
            # 只输出 accuracy
            average_accuracy = type_metrics[t]['accuracy'] / type_samples[t]
            final_results['type_metrics'][t]['accuracy'] = average_accuracy
            print(f"  accuracy: {average_accuracy:.2f}")
        elif t in {"ISVGEN", "TSVGEN"}:
            # 输出图像相关指标
            average_ssim = type_metrics[t]['ssim'] / type_samples[t]
            average_psnr = type_metrics[t]['psnr'] / type_samples[t]
            average_lpips = type_metrics[t]['lpips'] / type_samples[t]
            average_clip_score = type_metrics[t]['clip_score'] / type_samples[t]
            final_results['type_metrics'][t]['ssim'] = average_ssim
            final_results['type_metrics'][t]['psnr'] = average_psnr
            final_results['type_metrics'][t]['lpips'] = average_lpips
            final_results['type_metrics'][t]['clip_score'] = average_clip_score
            print(f"  ssim: {average_ssim:.2f}")
            print(f"  psnr: {average_psnr:.2f}")
            print(f"  lpips: {average_lpips:.2f}")
            print(f"  clip_score: {average_clip_score:.2f}")
        elif t in {"CSVGUN_usage","ISVGUN_usage","CSVGUN_rect","CSVGUN_circle","CSVGUN_description","ISVGUN_description"}:
            # 输出文本相关指标
            average_bertscore = type_metrics[t]['bertscore'] / type_samples[t]
            average_sbert_score = type_metrics[t]['sbert_score'] / type_samples[t]
            final_results['type_metrics'][t]['bertscore'] = average_bertscore
            final_results['type_metrics'][t]['sbert_score'] = average_sbert_score
            print(f"  bertscore: {average_bertscore:.2f}")
            print(f"  sbert_score: {average_sbert_score:.2f}")
    else:
        print(f"Type: {t}, No samples found")

# 计算ISVGEN和TSVGEN的综合得分
def calculate_isvgen_tsvgen_score(metrics):
    ssim = metrics['ssim'] * 0.2
    lpips = (1 - metrics['lpips']) * 0.2  # lpips值越小越好，因此取反
    clip_score = metrics['clip_score'] * 0.6
    return ssim + lpips + clip_score

isvgen_score = calculate_isvgen_tsvgen_score(final_results['type_metrics']['ISVGEN'])
tsvgen_score = calculate_isvgen_tsvgen_score(final_results['type_metrics']['TSVGEN'])

# 计算easy_acc
easy_acc = (
    final_results['type_metrics']['CSVGUN_size']['accuracy'] * 50 +
    final_results['type_metrics']['CSVGUN_shape']['accuracy'] * 50 +
    final_results['type_metrics']['CSVGUN_transform']['accuracy'] * 50
) / (50 + 50 + 50)

# 计算hard_acc
hard_acc = (
    (final_results['type_metrics']['ISVGUN_color']['accuracy'] + final_results['type_metrics']['CSVGUN_color']['accuracy']) / 2 * 100 +
    (final_results['type_metrics']['ISVGUN_category']['accuracy'] + final_results['type_metrics']['CSVGUN_category']['accuracy']) / 2 * 100
) / (100 + 100)

# 计算BERTScore和SBERT的综合得分
def calculate_bert_sbert_score(metrics_list):
    total_bertscore = 0
    total_sbert_score = 0
    total_weight = 0
    for metrics in metrics_list:
        weight = metrics['weight']
        total_bertscore += metrics['bertscore'] * weight
        total_sbert_score += metrics['sbert_score'] * weight
        total_weight += weight
    return total_bertscore / total_weight, total_sbert_score / total_weight

bert_sbert_metrics = [
    {'bertscore': final_results['type_metrics']['CSVGUN_usage']['bertscore'], 'sbert_score': final_results['type_metrics']['CSVGUN_usage']['sbert_score'], 'weight': 200},
    {'bertscore': final_results['type_metrics']['CSVGUN_description']['bertscore'], 'sbert_score': final_results['type_metrics']['CSVGUN_description']['sbert_score'], 'weight': 200},
    {'bertscore': final_results['type_metrics']['ISVGUN_usage']['bertscore'], 'sbert_score': final_results['type_metrics']['ISVGUN_usage']['sbert_score'], 'weight': 200},
    {'bertscore': final_results['type_metrics']['ISVGUN_description']['bertscore'], 'sbert_score': final_results['type_metrics']['ISVGUN_description']['sbert_score'], 'weight': 200},
    {'bertscore': final_results['type_metrics']['CSVGUN_rect']['bertscore'], 'sbert_score': final_results['type_metrics']['CSVGUN_rect']['sbert_score'], 'weight': 100},
    {'bertscore': final_results['type_metrics']['CSVGUN_circle']['bertscore'], 'sbert_score': final_results['type_metrics']['CSVGUN_circle']['sbert_score'], 'weight': 100}
]

bertscore, sbert_score = calculate_bert_sbert_score(bert_sbert_metrics)

# 计算最终得分
final_score = (
    isvgen_score * 0.45 +
    tsvgen_score * 0.45 +
    easy_acc * 0.01 +
    hard_acc * 0.02 +
    bertscore * 0.035 +
    sbert_score * 0.035
)

print(f"ISVGEN Score: {isvgen_score}")
print(f"TSVGEN Score: {tsvgen_score}")
print(f"Easy Accuracy (easy_acc): {easy_acc}")
print(f"Hard Accuracy (hard_acc): {hard_acc}")
print(f"BERTScore: {bertscore}")
print(f"SBERT Score: {sbert_score}")
print(f"Final Score: {final_score}")

# 保存最终结果为 JSON 文件
final_results['isvgen_score'] = isvgen_score
final_results['tsvgen_score'] = tsvgen_score
final_results['easy_acc'] = easy_acc
final_results['hard_acc'] = hard_acc
final_results['bertscore'] = bertscore
final_results['sbert_score'] = sbert_score
final_results['final_score'] = final_score

with open(output_json_path, 'w') as f:
    json.dump(final_results, f, indent=4)

print(f"Final results saved to {output_json_path}")
