# MitigV

A library of **training-free** hallucination mitigation algorithms for large
vision-language models (LVLMs). Instead of fine-tuning, these methods re-write
the decoding process (logits, attention, or sampling) at generation time.
Implemented algorithms: **VCD** (Visual Contrastive Decoding), **ICD**
(Instruction Contrastive Decoding) and **PAI** (Paying More Attention to Image).

## Quick start

```python
from mitigv import mitigate
from mitigv.algorithms.vcd import VCDConfig

with mitigate("vcd", VCDConfig(alpha=2.0), model=model, processor=processor,
              device="cuda:2") as f:
    text = f(images, prompt)
```

`mitigate` is a context manager: it builds the algorithm, yields a callable
mitigator, and on exit restores the model's device and frees CUDA cache.

## Design principles

- **Uniform API** — every algorithm exposes the same `generate(images, prompt, **kwargs)` entry point, so callers can swap algorithms without touching their code.
- **Inheritance & polymorphism** — all algorithms subclass `BaseMitigator`; the config system is a parallel hierarchy rooted at `MitigatorConfig`.
- **Small modules, each tested** — the library grows one module at a time, with tests written alongside.

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

### Module 3 — `mitigv.backends.hf`

The HuggingFace-transformers decoding skeleton. `HFMitigator` implements a
cache-aware autoregressive loop (greedy / sampling / **beam search**) and
exposes small hooks so an algorithm implements only its intervention:

```python
from mitigv import HFMitigator

class VCD(HFMitigator):
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
`HFMitigator` targets causal LMs / LVLMs with `use_cache=True`.

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
gamma_t   = exp(-lambda * t)
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

## Evaluation

`evaluators/` contains a POPE evaluator that runs the library against the VCD
paper's reported numbers (Table 1, LLaVA-1.5-7B) and judges whether differences
are ignorable:

```bash
PYTHONPATH=src python -m evaluators.run_pope --device cuda:2 --limit 500
```

`evaluators/pope.py` provides the reference table, metric computation (matching
the official `eval_pope.py`) and the `compare_to_reference` verdict helper.

## Layout

```
src/mitigv/
  __init__.py          # public exports
  core/
    __init__.py
    base.py            # module 1: MitigatorConfig + BaseMitigator
    registry.py        # module 2: registration + build_mitigator
  backends/
    __init__.py
    hf.py              # module 3: HFMitigator decoding skeleton
  perturbations.py     # module 4: image distortion operators
  algorithms/
    __init__.py
    vcd.py             # module 5: VCD (Visual Contrastive Decoding)
    icd.py             # module 6: ICD (Instruction Contrastive Decoding)
    pai.py             # module 7: PAI (Paying More Attention to Image)
    m3id.py            # module 8: M3ID (Multi-Modal Mutual Information Decoding)
    vista.py           # module 9: VISTA (Visual Information Steering)
  api.py               # mitigate() context manager
evaluators/
  pope.py              # POPE metrics + paper reference + verdict
  run_pope.py          # end-to-end POPE validation driver
tests/
  test_base.py
  test_registry.py
  test_hf.py
  test_perturbations.py
  test_vcd.py
  test_icd.py
  test_pai.py
  test_m3id.py
  test_vista.py
  test_beam.py
  test_pope.py
  test_mitigate.py
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```
