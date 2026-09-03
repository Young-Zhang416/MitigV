# MitigV

<p align="center">
  <img src="logo.png" alt="MitigV logo" width="520">
</p>

A library of **training-free** hallucination mitigation algorithms for large
vision-language models (LVLMs). Instead of fine-tuning, these methods re-write
the decoding process (logits, attention, or sampling) at generation time.
Implemented algorithms include **VCD**, **ICD**, **PAI**, **M3ID**, **VISTA**,
**AGLA**, **ONLY**, **OPERA**, and linear-probe steering.

## Quick start

Load a supported checkpoint and create an algorithm in one call:

```python
from mitigv import load_mitigator

vcd = load_mitigator(
    "vcd",
    model_type="qwen2.5-vl",  # or "llava"
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
    alpha=2.0,
)
text = vcd(image, "Describe the image.")
```

For already-loaded model and processor objects, use the context manager:

```python
from mitigv import mitigate
from mitigv.algorithms.vcd import VCDConfig

with mitigate("vcd", VCDConfig(alpha=2.0), model=model, processor=processor,
              device="cuda:2") as f:
    text = f(images, prompt)
```

`mitigate` is a context manager: it builds the algorithm, yields a callable
mitigator, and on exit restores the model's device and frees CUDA cache.

中文入门与完整示例请参阅 [Tutorial.md](Tutorial.md)。

## Installation

```bash
python -m pip install .                    # generic PyTorch backend
python -m pip install ".[transformers]"   # LLaVA and Qwen2.5-VL
python -m pip install ".[eval]"           # evaluation commands and dependencies
```

For development, use `python -m pip install -e ".[test,eval]"`.

## Design principles

- **Uniform API** — every algorithm exposes the same `generate(images, prompt, **kwargs)` entry point, so callers can swap algorithms without touching their code.
- **Inheritance & polymorphism** — all algorithms subclass `BaseMitigator`; the config system is a parallel hierarchy rooted at `MitigatorConfig`.
- **Small modules, each tested** — the library grows one module at a time, with tests written alongside.

## Implementation status and compatibility

VCD, ICD, PAI, and M3ID implement their published decoding equations. VISTA
currently implements the VSV component but not SLA. AGLA uses an internal
LVLM-attention crop as a lightweight approximation of the paper's separate
image-prompt matching/localization stage. LinearProbeSteer is a library-specific
representative rather than a reproduction of one named paper. ONLY preserves
the published TVER selection and adaptive fusion, but currently obtains the two
logit branches with two forwards rather than the paper's optimized single-query
implementation.

The attention-patching algorithms depend on Llama attention internals and are
tested against Transformers 5.x. PAI and ONLY therefore fail fast when required
image-token positions cannot be inferred. OPERA currently supports batch size 1
only because its rollback timeline is stateful; other algorithms support padded
batches and normalize token inputs to left padding.

## Current state

### Module 1 — `mitigv.core.base`

The two contracts everything else builds on.

### `MitigatorConfig`

Base hyper-parameter container (dataclass) with validation, copy and
(de)serialization. Algorithms subclass it to add their own knobs.

```python
from mitigv import MitigatorConfig, MitigatorConfigError

class VCDConfig(MitigatorConfig):   # no @dataclass needed — it is auto-applied
    alpha: float = 1.0
    beta: float = 0.1

    def validate(self):
        super().validate()
        if self.alpha < 0:
            raise MitigatorConfigError("alpha must be >= 0")

cfg = VCDConfig(alpha=2.0, max_new_tokens=64)
cfg.to_dict()          # -> {...}
VCDConfig.from_dict({...})   # rejects unknown keys
cfg.copy(alpha=3.0)    # new instance; original untouched
```

### `BaseMitigator`

Abstract base class. It owns the model + processor, resolves configuration, and
pins down the `generate` contract. Subclasses set `algorithm_name` and
`config_class`, then implement `generate`.

```python
from mitigv import BaseMitigator, MitigatorConfig

class MyMitigator(BaseMitigator):
    algorithm_name = "my_alg"
    config_class = MyConfig  # optional; defaults to MitigatorConfig

    def generate(self, images, prompt, **kwargs):
        self._ensure_ready()       # raise unless model + processor are set
        ...                        # intervention happens here
        return text
```

Construction accepts `config` as a config instance, a mapping, or `None`, and
treats any extra keyword argument as a validated config override:

```python
m = MyMitigator(model, processor, alpha=2.0)          # kwargs -> config
m = MyMitigator(model, processor, config={"alpha": 2}) # mapping -> config
m(images, prompt)                                      # __call__ == generate
```

### Module 2 — `mitigv.core.registry`

Name-based registration and instantiation — the mechanism that lets callers
switch algorithms polymorphically without importing concrete modules.

```python
from mitigv import build_mitigator, register_mitigator, list_mitigators

@register_mitigator("vcd")            # explicit key
class VCD(BaseMitigator): ...

@register_mitigator                   # bare form: uses cls.algorithm_name
class PAI(BaseMitigator):
    algorithm_name = "pai"
    ...

list_mitigators()                          # -> ['pai', 'vcd']
m = build_mitigator("vcd", model, processor, alpha=2.0)  # config/kwargs forwarded
text = m(images, prompt)
```

Registration is idempotent for the same class under the same name; a *different*
class under an existing name raises unless `override=True` is passed.

### Module 3 — `mitigv.backends`

The cache-aware autoregressive decoding skeleton. `GenericMitigator` (with
`ModelMitigator` as a convenience alias) implements greedy /
sampling / **beam search** and exposes small hooks so an algorithm implements
only its intervention. It is framework-neutral: HuggingFace is one supported
implementation, not a requirement.

Any model and processor satisfying the following structural interfaces can be
used:

* `ModelProtocol`: callable with `input_ids`, optional `attention_mask`,
  `past_key_values`, and `use_cache=True`; returns `logits` and optionally
  `past_key_values` (as attributes, mapping keys, or tuple items).
* `ProcessorProtocol`: callable with `text=...`, `images=...` returning a
  mapping containing `input_ids`, and providing `batch_decode` (or `decode`).
  Processors may alternatively expose `prepare_inputs(prompt, images)` or
  `encode(prompt, images)`.

The protocols are typing contracts (`ModelInterface` and
`ProcessorInterface` are aliases), so no HuggingFace base class or inheritance
is needed.

```python
from mitigv import ModelMitigator

class VCD(ModelMitigator):
    algorithm_name = "vcd"

    def _step_logits(self, input_ids, attention_mask, inputs, past, step, cfg):
        logits_v, past = self._forward(input_ids, attention_mask, inputs, past)
        logits_vp, _ = self._forward(input_ids, attention_mask, self._distorted(inputs), None)  # sketch
        return (1 + self.config.alpha) * logits_v - self.config.alpha * logits_vp, past
```

Hooks (all overridable):

| hook | responsibility |
|------|----------------|
| `_prepare_inputs(images, prompt, cfg)` | build the model-input tensor dict |
| `_step_logits(..., cfg)` | per-step (possibly intervened) next-token logits |
| `_forward(...)` | one raw forward pass on the main model |
| `_sample_next(logits, cfg)` | greedy / temperature / top-k / top-p sampling |
| `_reorder_cache` / `_reorder_aux_cache` | beam-search cache reordering |
| `_decode(generated_ids)` | ids → `str` (or `list[str]` for a batch) |
| `_eos_token_id()` | resolve the EOS token id |

Beam search is enabled with `num_beams > 1` and honors `length_penalty`,
`early_stopping` and `num_return_sequences`. Visual inputs (`pixel_values`, ...)
are forwarded only on the first step; the rest of the loop is KV-cache-aware.
`ModelMitigator` targets causal/autoregressive models with `use_cache=True`.
Algorithms that inspect decoder-layer attention or hidden states (for example
`PAI`, `ONLY`, `AGLA`, and `OPERA`) additionally require the corresponding
optional hooks exposed by the model; the basic decoding and contrastive
algorithms only use the interfaces above.

For HuggingFace LLaVA, use `adapt_llava(model, processor)` from
`mitigv.backends.llava`; all Transformers-specific handling stays in the
adapter while algorithms remain backend-independent:

```python
from mitigv import build_mitigator
from mitigv.backends.llava import adapt_llava

model, processor = adapt_llava(model, processor)
vcd = build_mitigator("vcd", model=model, processor=processor)
text = vcd(image, "Describe the image.")
```
Adapters also expose lazy `from_pretrained(checkpoint)` factories when loading
directly from Transformers is desired.

Qwen2.5-VL is supported through `mitigv.backends.qwen2_5_vl`. Its adapter
handles the Qwen chat template and preserves multimodal fields such as
`image_grid_thw`, `video_grid_thw`, `mm_token_type_ids`, and `rope_deltas`.

LLaVA and Qwen2.5-VL share the `VisionLanguageModelAdapter` and
`VisionLanguageProcessorAdapter` parent contracts. Select the concrete model
family through one parameter:

```python
from mitigv import load_vision_language

model, processor = load_vision_language(
    model_type="qwen2.5-vl",  # or "llava"
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    model_kwargs={"torch_dtype": "auto"},
)
```

For already-loaded objects use `adapt_vision_language(model_type, model,
processor)`. Supported aliases include `qwen`, `qwen2_5_vl`, `llava-next`, and
`llava-1.5`.

### CHAIR Evaluation

`mitigv.evaluation.chair` provides a strict, deterministic CHAIR evaluator
using the bundled official synonym table. It reports per-image object details
and 1000-resample image-level 95% bootstrap intervals for `CHAIRs`, `CHAIRi`,
object recall, object F1, and mean word/sentence length. COCO files are read
locally; the CLI defaults to `~/dataset/coco2017/annotations` and never
downloads data:

```bash
python -m mitigv.evaluation.chair \
  --generated-json outputs/captions.json \
  --output-json outputs/chair.json
```

Prediction items must contain `image_id` and one of `caption`,
`generated_text`, `text`, `answer`, or `output`. COCO ground truth is the union
of instance categories and recognized objects in the reference captions.

The supplementary double judge is available as `mitigv.evaluation.judge`. It calls
DeepSeek (`deepseek-chat`, temperature 0, JSON response format) using the fixed
prompt `mitigv/evaluation/prompts/extract_objects.txt`, caches by caption SHA-256, and
verifies each extracted noun phrase with one resident GroundingDINO service.
The default local COCO paths are under `~/dataset/coco2017`; it writes
`results/judge.json` and a 500-image `results/judge_audit_sample.jsonl`.

### Discriminative Evaluation

`mitigv.evaluation.discriminative` evaluates POPE's `random`, `popular`, and
`adversarial` subsets plus AMBER discriminative parquet files. It parses the
first standalone `yes`/`no` token (including `Yes,` and `No.`), reports
accuracy, precision, recall, F1, and emits one JSON detail row per question.
`mitigv.evaluation.length_analysis` fits a Poisson model of image-level hallucination
counts with description word count as a covariate, reports length-adjusted
CHAIRi residual gains against a baseline configuration, and emits
`length_chairi_scatter` points for plotting.

The local AMBER discriminative files are under `~/dataset/AMBER`, for example
`discriminative-existence-00000-of-00001.parquet`; predictions are supplied in
the same row order as the parquet records.

### Module 4 — `mitigv.perturbations`

Image distortion operators, with a name-based registry so configs stay
serializable and new distortions can be registered:

```python
from mitigv.perturbations import build_perturbation, list_perturbations

list_perturbations()                        # -> ['diffusion_noise', 'gaussian_noise']
p = build_perturbation("gaussian_noise", std=0.1)
distorted = p(pixel_values)                 # same shape/dtype, noise added
```

* `GaussianNoisePerturbation(std, clip=None)` — `image + std * N(0, I)`.
* `DiffusionNoisePerturbation(noise_step, num_steps=1000)` — DDPM forward-process
  noise (the distortion used by VCD).

### Module 5 — `mitigv.algorithms.vcd`

**VCD** (Visual Contrastive Decoding). At each step it runs the model on the
original and a distorted image, then contrasts the logits, with an optional
adaptive plausibility constraint:

```
logits = (1 + alpha) * logits(v) - alpha * logits(v')
# + adaptive plausibility: mask tokens whose *original*-branch probability
#   < beta * max  (equivalently, logits(v) < log(beta) + max(logits(v)))
```

```python
from mitigv import build_mitigator

vcd = build_mitigator("vcd", model, processor, alpha=1.0, beta=0.1)
text = vcd(images, prompt)
```

`VCDConfig` adds `alpha` (contrast strength), `beta` (plausibility threshold),
`distortion` (perturbation name) and `distortion_kwargs`.

### Module 6 — `mitigv.algorithms.icd`

**ICD** (Instruction Contrastive Decoding, Wang et al., ACL Findings 2024).
Like VCD it is a dual-branch contrastive decoder, but the disturbance is applied
to the *instruction* (a role prefix, e.g. "You are a confused object detector ...")
rather than the image:

```
logits = logits(std) - lam * logits(disturbed)
# + adaptive plausibility: mask tokens whose *standard*-branch probability
#   < alpha * max
```

```python
from mitigv import build_mitigator

icd = build_mitigator("icd", model, processor, lam=1.0, alpha=0.1)
text = icd(images, prompt)
```

`ICDConfig` adds `lam` (contrast strength, λ=1.0), `alpha` (plausibility
threshold, α=0.1) and `disturbance_prefix`.

### Module 7 — `mitigv.algorithms.pai`

**PAI** (Paying More Attention to Image, Liu et al., ECCV 2024). Two coordinated
interventions: it amplifies the image-token attention of the newest query token
during decoding (weight `alpha`), and contrasts the logits against a *text-only*
branch — the same prompt with the image removed — to counter the language prior
("text inertia"):

```
A[:, :, -1, img_start:img_end] = |A[...]| * alpha + A[...]   # pre-softmax, layers [start_layer, end_layer)
logits = gamma * (logits(image) - logits(text)) + logits(text)
# + adaptive plausibility: mask tokens whose *with-image* probability < beta * max
```

```python
from mitigv import build_mitigator

pai = build_mitigator("pai", model, processor, alpha=0.2, gamma=1.1)
text = pai(images, prompt)
```

`PAIConfig` adds `alpha` (attention scale), `gamma` (guidance scale, `1` disables
the text guidance), `beta` (plausibility threshold), `start_layer`/`end_layer`
(attention-intervention layer range) and `num_image_tokens` (auto-detected when
`None`). The attention intervention targets Llama-style self-attention and is
disabled with `alpha=0`.

### Module 8 — `mitigv.algorithms.m3id`

**M3ID** (Multi-Modal Mutual Information Decoding, Favero et al., CVPR 2024).
Counteracts "conditioning dilution" by contrasting the with-image branch against
a no-image branch with a weight that *grows* over decoding steps, gated by the
model's confidence:

```
gamma_t   = exp(-lambda * t)  # t=1 for the first predicted token
weight_t  = (1 - gamma_t) / gamma_t
logits    = l_c + 1[max(l_c) < log(alpha)] * weight_t * (l_c - l_u)
```

```python
from mitigv import build_mitigator

m3id = build_mitigator("m3id", model, processor, alpha=0.3, forgetting_rate=0.02)
text = m3id(images, prompt)
```

`M3IDConfig` adds `alpha` (plausibility threshold, default 0.3; scan 0.2/0.3/0.5)
and `forgetting_rate` (`lambda`, default 0.02; scan 0.001/0.02/0.03).

### Module 9 — `mitigv.algorithms.vista`

**VISTA** (Visual Information Steering, Li & Shi, ICML 2025). Counteracts the
dilution of visual information in the residual stream: it extracts a per-layer
Visual Steering Vector (VSV) as the difference of the residual streams of the
with-image and no-image prompts, then injects it into every layer during
decoding:

```
V_steer^l = F(X_p)^l[last] - F(X_n)^l[last]
h_t^l     = h_t^l + steer_strength * V_steer^l
```

```python
from mitigv import build_mitigator

vista = build_mitigator("vista", model, processor, steer_strength=0.01)
text = vista(images, prompt)
```

`VISTAConfig` adds `steer_strength` (λ, default 0.01). Only the VSV component is
implemented; the paper's logits-level SLA component is left as future work.

### Module 10 — `mitigv.algorithms.probe_steer`

**LinearProbeSteer** — a self-implemented representative of the representation
steering family. A linear probe is trained (on frozen features) to classify
object presence/absence; its decision normal steers the residual stream:

```
h_t^layer = h_t^layer + beta * (w / ||w||)
```

```python
import torch
from mitigv import build_mitigator

probe = torch.load("probe.pt")
steer = build_mitigator(
    "linear_probe_steer", model, processor,
    steering_vector=probe["weight"], layer=probe["layer"], beta=5.0,
)
text = steer(images, prompt)
```

`LinearProbeSteerConfig` adds `beta` (injection strength; scan `{2, 5, 8, 12}`)
and `layer` (the decoder layer the probe was trained on). The probe is fit by
`mitigv-train-probe` on ~2000 COCO images (~30 min, single GPU); no model
weight is updated.

### Module 11 — `mitigv.algorithms.agla`

**AGLA** (Assembly of Global and Local Attention, An et al., 2025). Fuses the
original image's generative global view with a saliency-cropped local view's
discriminative logits:

```
logits = logit(global) + alpha * logit(local)
```

```python
from mitigv import build_mitigator

agla = build_mitigator("agla", model, processor, alpha=1.0, crop_ratio=0.5)
text = agla(images, prompt)
```

`AGLAConfig` adds `alpha` (local-logits weight, default 1.0) and `crop_ratio`
(crop side as a fraction of the shorter edge, default 0.5). The saliency map is
estimated from the LVLM's own attention to image tokens (eager attention is
forced for the probe, then restored).

### Module 12 — `mitigv.algorithms.only`

**ONLY** (One-Layer Intervention, Wan et al., ICCV 2025). At one decoder layer,
attention heads are ranked by their Text-to-Visual Entropy Ratio (TVER); heads
below the layer average are deactivated to produce a textually-enhanced logits
distribution, which is then fused with the original adaptively:

```
d     = sum_y |p(y) - p~(y)|
final = f + alpha1 * f~                    if d < gamma   (collaborative)
final = (1 + alpha2) * f - alpha2 * f~     otherwise      (contrastive)
```

```python
from mitigv import build_mitigator

only = build_mitigator("only", model, processor, layer=0, alpha1=3.0, alpha2=1.0, gamma=0.2)
text = only(images, prompt)
```

`ONLYConfig` adds `layer`, `alpha1` (=3), `alpha2` (=1) and `gamma` (=0.2).

### Module 13 — `mitigv.algorithms.opera`

**OPERA** (Over-trust Penalty and Retrospection-Allocation, Huang et al., CVPR 2024).
Beam-search decoding that detects the "knowledge aggregation" (columnar
attention) pattern and penalizes it, then rolls back and re-selects when the
pattern persists:

```
phi     = max_j  prod_{i=j}^{t-1} sigma * omega[i, j]   # raw column product, FP32
score   = log_softmax(logits) + beam_score - penalty_weight * phi   # candidate-level
```

```python
from mitigv import build_mitigator

opera = build_mitigator("opera", model, processor, num_beams=5)
text = opera(images, prompt)
```

`OPERAConfig` adds `num_beams` (=5), `sigma` (=50), `penalty_weight` (=1),
`num_attn_candidates` (=5, top candidates for the cached look-ahead),
`window_size` (=5), `threshold` (=15, overlap count for retrospection),
`retrospection_window` (=20) and `max_rollback` (=30). Attention uses max-over-heads
then re-normalization; the penalty is computed on each candidate's own attention
via a cached look-ahead forward, and retrospection performs a real rollback
(snapshot restore + old-candidate exclusion + re-selection).

## Evaluation

`mitigv.evaluation` contains a POPE evaluator that runs the library against the VCD
paper's reported numbers (Table 1, LLaVA-1.5-7B) and judges whether differences
are ignorable:

```bash
mitigv-pope --device cuda:2 --limit 500
```

`mitigv/evaluation/pope.py` provides the reference table, metric computation (matching
the official `eval_pope.py`) and the `compare_to_reference` verdict helper.

## Layout

```
src/mitigv/
  __init__.py          # public exports
  core/
    __init__.py
    base.py            # module 1: MitigatorConfig + BaseMitigator
    interfaces.py      # structural model/processor contracts
    registry.py        # module 2: registration + build_mitigator
  backends/
    __init__.py
    generic.py         # framework-neutral decoding backend
    factory.py         # model-family selection and loading
    hf_common.py       # shared Transformers adapter parents
    llava.py           # HuggingFace LLaVA adapters
    qwen2_5_vl.py      # HuggingFace Qwen2.5-VL adapters
  perturbations.py     # module 4: image distortion operators
  algorithms/
    __init__.py
    vcd.py             # module 5: VCD (Visual Contrastive Decoding)
    icd.py             # module 6: ICD (Instruction Contrastive Decoding)
    pai.py             # module 7: PAI (Paying More Attention to Image)
    m3id.py            # module 8: M3ID (Multi-Modal Mutual Information Decoding)
    vista.py           # module 9: VISTA (Visual Information Steering)
    probe_steer.py     # module 10: LinearProbeSteer (probe-normal steering)
    agla.py            # module 11: AGLA (Assembly of Global and Local Attention)
    only.py            # module 12: ONLY (One-Layer Intervention)
    opera.py           # module 13: OPERA (Over-trust Penalty + Retrospection)
  evaluation/
    chair.py           # strict CHAIR + bootstrap confidence intervals
    pope.py            # POPE metrics + paper reference
    discriminative.py  # POPE + AMBER yes/no evaluation
    length_analysis.py # Poisson length-control analysis
    judge.py           # DeepSeek + GroundingDINO supplementary judge
    train_probe.py     # linear object-presence probe training
    data/               # packaged CHAIR synonyms
    prompts/            # packaged evaluator prompts
  api.py               # load_mitigator() + mitigate()
tests/
  test_base.py
  test_registry.py
  test_generic.py
  test_perturbations.py
  test_vcd.py
  test_icd.py
  test_pai.py
  test_m3id.py
  test_vista.py
  test_probe_steer.py
  test_agla.py
  test_only.py
  test_opera.py
  test_beam.py
  test_pope.py
  test_mitigate.py
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```
