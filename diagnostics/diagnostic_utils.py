"""
Diagnostic Utilities — 共享工具层
提供模型加载、文本生成、Gemini评分等基础能力，供所有模块使用。
"""
import os
import sys
import json
import time
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

# ── 路径 ──────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
TOKENIZER_PATH = os.path.join(ROOT_DIR, 'model')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'raw')
FIGURES_DIR = os.path.join(ROOT_DIR, 'results', 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── LLM 配置 ──────────────────────────────────────────
STAGES = {
    'pretrain': 'pretrain_768',
    'sft': 'full_sft_768',
    'grpo': 'grpo_768',
    'dpo': 'dpo_768',
}

# ── VLM 配置 ──────────────────────────────────────────
VLM_STAGES = {
    'vlm_pretrain': 'vlm_pretrain_768',
    'vlm_sft': 'vlm_sft_768',
}

ALL_STAGES = {**STAGES, **VLM_STAGES}
LLM_STAGE_ORDER = ['pretrain', 'sft', 'grpo', 'dpo']
VLM_STAGE_ORDER = ['vlm_pretrain', 'vlm_sft']
FULL_STAGE_ORDER = LLM_STAGE_ORDER + VLM_STAGE_ORDER

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def vlm_checkpoints_available():
    """检查 VLM checkpoint 是否存在"""
    for ckpt_name in VLM_STAGES.values():
        if not os.path.exists(os.path.join(CHECKPOINT_DIR, f'{ckpt_name}.pth')):
            return False
    return True


def save_figure(fig, fig_name):
    """统一保存图表到 FIGURES_DIR，使用 v5 命名"""
    path = os.path.join(FIGURES_DIR, fig_name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# ── 模型加载 ──────────────────────────────────────────
def load_model(ckpt_name, flash_attn=True, device=None):
    """加载 MiniMind LLM checkpoint"""
    device = device or DEVICE
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, flash_attn=flash_attn)
    model = MiniMindForCausalLM(config)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{ckpt_name}.pth')
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.half().eval().to(device)
    return model


def load_tokenizer():
    """加载 tokenizer"""
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


# ── VLM 模型加载 ──────────────────────────────────────
def load_vlm_model(ckpt_name, device=None):
    """加载 MiniMind VLM checkpoint"""
    device = device or DEVICE
    from model.model_vlm import VLMConfig, MiniMindVLM
    config = VLMConfig(hidden_size=768, num_hidden_layers=8)
    vision_model_path = os.path.join(ROOT_DIR, 'model', 'siglip2-base-p16-ve')
    model = MiniMindVLM(config, vision_model_path=vision_model_path)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{ckpt_name}.pth')
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.half().eval().to(device)
    return model


@torch.no_grad()
def generate_vlm_response(model, tokenizer, prompt, image,
                          max_new_tokens=256, temperature=0.7, top_p=0.9):
    """VLM 生成：文本 + 图片输入"""
    from PIL import Image
    input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)

    # 处理图片 — 传递原始 processor 输出, forward() 内部完成 vision encode + proj
    if model.processor is not None and image is not None:
        image_inputs = model.image2tensor(image, model.processor)
        pixel_values = {k: v.to(model.device) for k, v in image_inputs.items()}
    else:
        pixel_values = None

    output_ids = model.generate(
        input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── 文本生成 ──────────────────────────────────────────
@torch.no_grad()
def generate_response(model, tokenizer, prompt, stage_name,
                      max_new_tokens=256, temperature=0.7, top_p=0.9):
    """生成单条回复，根据阶段选择不同输入格式"""
    if stage_name == 'pretrain':
        input_text = tokenizer.bos_token + prompt
    else:
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )

    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Gemini 评分 ──────────────────────────────────────
def _get_gemini_client():
    """获取 Gemini API client"""
    try:
        from google import genai
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print('  [WARN] GEMINI_API_KEY not set in environment. Gemini features disabled.')
            return None
        return genai.Client(api_key=api_key)
    except ImportError:
        return None


def call_gemini(prompt_text, max_retries=3):
    """调用 Gemini API，返回文本回复"""
    client = _get_gemini_client()
    if client is None:
        return None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_text
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [Gemini API error after {max_retries} retries]: {e}")
                return None


def gemini_judge(question, response, criterion):
    """用 Gemini 做单维度判断（返回文本判断结果）"""
    prompt = f"""请根据以下标准判断AI模型的输出。

用户问题：{question}
AI输出：{response}
评估标准：{criterion}

请简洁回答。"""
    return call_gemini(prompt)


def gemini_score(question, response, dimension, scale='1-5'):
    """用 Gemini 做单维度打分（返回数值）"""
    prompt = f"""你是LLM输出质量评估专家。请对以下模型回复进行{dimension}维度评分。

用户问题：{question}
模型回复：{response}

请给出{scale}分的整数评分。只回答一个数字。"""
    result = call_gemini(prompt)
    if result:
        nums = re.findall(r'\d+', result)
        if nums:
            return int(nums[0])
    return None


# ── 工具函数 ──────────────────────────────────────────
def save_json(data, filename):
    """保存 JSON 到顶层 results/raw 目录"""
    path = os.path.join(RESULTS_DIR, filename)

    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, torch.Tensor):
                return None  # skip tensors
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=_Encoder)
    print(f"  Saved: {path}")


def load_json(filename):
    """从顶层 results/raw 目录加载 JSON"""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_test_images(max_images=15):
    """加载 VLM 测试图片，带 ground truth 描述、QA 和 distractor 标注。

    返回 list[dict]，每个 dict 包含:
        id, image (PIL.Image), description, qa (list), distractor
    """
    from PIL import Image

    images_dir = os.path.join(ROOT_DIR, 'images')
    annotations_path = os.path.join(os.path.dirname(__file__), 'vlm_test_data.json')

    # 加载标注文件
    annotations = {}
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for entry in json.load(f):
                annotations[entry['id']] = entry

    test_images = []
    if os.path.isdir(images_dir):
        for fname in sorted(os.listdir(images_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if fname.lower().endswith('.gif'):
                    continue
                img_path = os.path.join(images_dir, fname)
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception:
                    continue
                ann = annotations.get(fname, {})
                desc = ann.get('description', '') or f'一张名为{fname}的图片'
                test_images.append({
                    'id': fname,
                    'image': img,
                    'description': desc,
                    'qa': ann.get('qa', []),
                    'distractor': ann.get('distractor', '一群企鹅在沙漠中行走'),
                })
                if len(test_images) >= max_images:
                    break
    return test_images


def print_header(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def print_table(headers, rows):
    """打印简单对齐表格"""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = '─┼─'.join('─' * w for w in widths)
    hdr = ' │ '.join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(hdr)
    print(sep)
    for row in rows:
        print(' │ '.join(str(c).ljust(w) for c, w in zip(row, widths)))
