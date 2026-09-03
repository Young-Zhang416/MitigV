# MitigV 中文使用手册

本教程介绍如何在不微调模型的情况下，使用 MitigV 减少视觉语言模型（LVLM）的幻觉。示例均基于本仓库当前 API；模型权重和 COCO 数据集不会由脚本自动下载。

## 1. 安装与环境

建议使用 Python 3.10 及以上，并在独立虚拟环境中安装：

```bash
python -m pip install -e ".[transformers]"  # LLaVA/Qwen 推理
python -m pip install -e ".[eval]"          # 完整评测环境
python -m pip install -e ".[test,eval]"     # 贡献者开发环境
python -m pytest -q                          # 验证安装
ruff check .                                 # 代码风格检查
```

仅使用自定义 PyTorch 模型时，执行 `python -m pip install -e .` 即可。COCO、POPE 和 AMBER 数据应放在 `~/dataset` 下。

## 2. 加载模型并生成描述

MitigV 将模型和处理器抽象为两个小接口：模型能够接收 `input_ids` 并返回 `logits`，处理器能够把文本/图像编码为映射并提供 `batch_decode`。因此，除 HuggingFace 外的运行时也可以直接接入。

### 使用 LLaVA 或 Qwen2.5-VL

通过 `model_type` 选择模型族，算法代码无需改动：

```python
from PIL import Image
from mitigv import load_mitigator

vcd = load_mitigator(
    "vcd",
    model_type="qwen2.5-vl",                 # 也可写 "llava"
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
    alpha=1.0,
    beta=0.1,
)
image = Image.open("example.jpg").convert("RGB")
prompt = "Describe the image in one sentence."
caption = vcd(image, prompt)
print(caption)
```

Qwen 适配器会自动处理 chat template；LLaVA 通常需要在提示中包含 `<image>` 占位符。已加载的原生对象可使用 `adapt_vision_language("llava", model, processor)`，别名包括 `qwen`、`qwen2_5_vl`、`llava-next` 和 `llava-1.5`。

### 使用上下文管理器

`mitigate` 会在退出时恢复模型设备并清理 CUDA 缓存：

```python
from mitigv import mitigate
from mitigv.algorithms.vcd import VCDConfig

with mitigate("vcd", VCDConfig(alpha=2.0), model=model,
              processor=processor, model_type="qwen2.5-vl", device="cuda:0") as decoder:
    caption = decoder(image, prompt, max_new_tokens=64)
```

对于直接加载的 HuggingFace LLaVA/Qwen 对象，`load_mitigator` 和
`mitigate` 也会根据 `model.config.model_type` 自动选择适配器；显式传入
`model_type` 可避免包装自定义模型时的歧义。

批量输入可传入图像列表和提示词列表；返回值为字符串列表。`num_beams>1` 开启 beam search，`do_sample=True` 开启温度、top-k 或 top-p 采样。设置 `seed` 可复现实验。

## 3. 选择和调节算法

所有算法都通过 `build_mitigator(name, model, processor, **kwargs)` 创建，常用名称如下：

| 名称 | 作用 | 主要参数 |
| --- | --- | --- |
| `vcd` | 原图与扰动图对比解码 | `alpha`、`beta`、`distortion` |
| `icd` | 正常指令与干扰指令对比 | `lam`、`alpha`、`disturbance_prefix` |
| `pai` | 放大图像 token 注意力并抑制文本先验 | `alpha`、`gamma`、`beta` |
| `m3id` | 随生成步数增强视觉分支 | `alpha`、`forgetting_rate` |
| `vista` | 注入视觉 steering vector | `steer_strength` |
| `agla` | 全局/局部图像双分支融合 | `alpha`、`crop_ratio` |
| `only` | 单层 TVER 头干预 | `layer`、`alpha1`、`alpha2` |
| `opera` | 过度信任惩罚与回溯 beam search | `num_beams`、`sigma` |
| `linear_probe_steer` | 线性探针表示引导 | `steering_vector`、`layer`、`beta` |

参数会经过配置校验；未知参数或非法范围会抛出 `MitigatorConfigError`。可用 `mitigv.list_mitigators()` 查看已注册算法。

## 4. 接入自定义模型

自定义运行时无需继承 HuggingFace 类。实现可调用模型和处理器即可：

```python
class MyProcessor:
    def __call__(self, *, text, images=None, return_tensors="pt", **kw):
        return {"input_ids": ..., "attention_mask": ...}
    def batch_decode(self, sequences, **kw):
        return ["..." for _ in sequences]

class MyModel:
    def __call__(self, **kw):
        return type("Output", (), {"logits": ..., "past_key_values": ...})()

decoder = build_mitigator("vcd", MyModel(), MyProcessor())
```

模型应支持 `use_cache=True` 和 `past_key_values`；图像张量只在首个解码步传入。需要访问注意力或隐藏状态的 PAI、ONLY、AGLA、OPERA、VISTA 还要求模型暴露相应层结构。

## 5. 评测流程

### 严格 CHAIR

输入 JSON 每项至少包含 `image_id` 和 `caption`（也接受 `generated_text`、`text`、`answer` 或 `output`）：

```bash
mkdir -p outputs results
python -m mitigv.evaluation.chair \
  --generated-json outputs/captions.json \
  --output-json results/chair.json \
  --instances-json ~/dataset/coco2017/annotations/instances_val2017.json \
  --captions-json ~/dataset/coco2017/annotations/captions_val2017.json
```

结果包含每图 `mentioned_objects`、`hallucinated`、`recalled`、`missed`，以及 CHAIRs、CHAIRi、object recall/F1、平均词数/句数和 1000 次图像级 bootstrap 的 95% CI。

### 使用样本 JSON 自动生成并评测

对于包含 `image_id`、`file_name` 和 `gt_objects` 的样本文件，可以自动读取图像、
调用 pipeline 生成描述并运行严格 CHAIR：

```json
[
  {"image_id": 184613, "file_name": "COCO_val2014_000000184613.jpg",
   "gt_objects": ["cow", "person", "umbrella"]}
]
```

```python
from mitigv.evaluation.pipeline import evaluate_pipeline_json

result = evaluate_pipeline_json(
    "samples.json", vcd, image_root="~/dataset",
    output_json="results/chair.json",
)
```

`gt_objects` 会直接作为每图的 COCO 对象集合，不需要额外 annotations；允许为空列表，
表示该图没有可用的 GT 对象，此时描述中出现的物体都会计为 hallucinated。也可以用
CLI 从本地 checkpoint 加载模型：

```bash
mitigv-evaluate-pipeline --input-json samples.json --image-root ~/dataset \
  --model-type llava --model-id ~/checkpoints/llava-1.5-7b-hf \
  --output-json results/chair.json
```

### DeepSeek + GroundingDINO 补充裁判

先设置 `DEEPSEEK_API_KEY`，再运行：

```bash
mitigv-judge \
  --generated-json outputs/captions.json \
  --output-json results/judge.json \
  --audit-path results/judge_audit_sample.jsonl
```

抽取结果按 caption 的 SHA-256 缓存；GroundingDINO 置信度大于 0.35 才算视觉可证实。审计文件固定输出图像路径、caption、抽取物体和 DINO 判定，供人工复核。

### POPE、AMBER 与长度校正

```bash
mitigv-discriminative \
  --questions ~/dataset/POPE/coco_pope_random.json \
  --predictions results/random_predictions.json

mitigv-length-analysis \
  --config baseline results/baseline_chair.json \
  --config vcd results/vcd_chair.json \
  --baseline baseline
```

判别式评测解析 `Yes`/`No` 变体并输出 accuracy、precision、recall、F1 和逐题明细；长度分析以词数为协变量拟合泊松模型，报告长度校正后的 CHAIRi 残差增益及散点数据。

### 参数自动调优

使用调参集（格式同上）和参数网格自动选择最优配置。每组参数都会重新创建
mitigator，但模型权重只加载一次：

```python
from mitigv import load_mitigator, tune_mitigator

base = load_mitigator(
    "vcd", model_type="llava", model_id="~/checkpoints/llava-1.5-7b-hf",
    max_new_tokens=64,
)
result = tune_mitigator(
    "tuning.json", algorithm="vcd", model=base.model, processor=base.processor,
    base_config=base.config,
    param_grid={"alpha": [0.5, 1.0, 1.5], "beta": [0.0, 0.1]},
    image_root="~/dataset", metric="CHAIRi",
    output_json="results/tuning.json",
)
print(result["best_params"], result["best_score"])
```

`CHAIRi` 和 `CHAIRs` 默认取最小值，`object_f1`、`object_recall` 默认取最大值；
也可以显式传入 `maximize=True` 或 `False`。命令行支持内联 JSON 或网格文件：

```bash
mitigv-tune --input-json tuning.json \
  --param-grid '{"alpha":[0.5,1.0],"beta":[0,0.1]}' \
  --algorithm vcd --model-type qwen2.5-vl \
  --model-id ~/checkpoints/qwen2.5-vl-7b-instruct \
  --output-json results/tuning.json
```

非 MitigV 运行时可传入 `pipeline_factory(params)`，由工厂返回可调用的图像描述
pipeline；其余评测和结果格式不变。

### 生成论文前沿图

先用 `tune_mitigator(..., metric="object_f1")` 在第 501--800 张调参图上为每个
发表方法选出配置；平凡基线的所有配置直接分别运行，不要调用调参器。把每个结果
文件写入一个 manifest（每行一个 JSON 对象）：

```json
{"name":"temperature-0.7","kind":"baseline","dataset":"coco_val500","model":"llava","result":"coco/temp07.json"}
{"name":"vcd","kind":"published","family":"contrastive","dataset":"coco_val500","model":"llava","result":"coco/vcd.json"}
```

然后运行：

```bash
python analysis/frontier.py --manifest results/expA_manifest.jsonl \
  --output-dir results/expA --bootstrap-samples 1000 --seed 42
```

脚本按 `dataset/model` 自动生成 COCO、AMBER、LLaVA、Qwen 的分组结果，输出基线
帕累托阶梯线及 bootstrap 置信带、发表方法置信椭圆、配对 bootstrap 的 Holm 校正
结论和 `expA_report.md`。安装绘图依赖：
`python -m pip install -e ".[eval,analysis]"`。

## 6. 实验建议与排错

固定随机种子、记录完整配置和模型 checkpoint；比较算法时保持图像顺序及生成长度一致。若出现显存不足，降低 `max_new_tokens`、batch size 或 beam 数。若适配器提示缺少 `transformers`，确认安装了 `python -m pip install -e ".[transformers]"`。评测脚本只读取本地数据，API 密钥应通过环境变量提供，切勿提交到仓库。
