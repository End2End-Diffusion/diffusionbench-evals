# diffusionbench-evals

Eval shims for [DiffusionBench](https://github.com/End2End-Diffusion/diffusion-bench), updated from [nanovisionx/nanogen-evals](https://github.com/nanovisionx/nanogen-evals). Four independent image generation evaluators consolidated into one repo, each installable as its own package.

| Subdir | Package | Metric | Upstream |
|---|---|---|---|
| `dpgbench/` | `dpg_evaluator` | DPG-Bench (VQA-based composition) | [TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA) |
| `geneval/` | `geneval_evaluator` | GenEval (object presence, count, color, position) | [djghosh13/geneval](https://github.com/djghosh13/geneval) |
| `t2v_metrics/` | `t2v_metrics` | VQAScore (image-text alignment), with both Qwen3.5 and Qwen3.6 series backends | [linzhiqiu/t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) |
| `fd_evaluator/` | `fd_evaluator` | FID / FDR6 / MIND across six representation spaces | [Jiawei-Yang/FD-loss](https://github.com/Jiawei-Yang/FD-loss), [toshas/torch-fidelity](https://github.com/toshas/torch-fidelity) |

Each fork strips upstream to evaluation-only code, removes heavy deps (mmdet, modelscope, video stack), and standardizes on `pyproject.toml` + uv. All packages now require Hugging Face Transformers 5.3.0 (up from the v4 series). Numerical scores match the originals.

## Install

Via uv with `subdirectory`:

```toml
[tool.uv.sources]
dpg-evaluator     = { git = "ssh://git@github.com/End2End-Diffusion/diffusionbench-evals.git", subdirectory = "dpgbench",    branch = "main" }
geneval-evaluator = { git = "ssh://git@github.com/End2End-Diffusion/diffusionbench-evals.git", subdirectory = "geneval",     branch = "main" }
t2v-metrics       = { git = "ssh://git@github.com/End2End-Diffusion/diffusionbench-evals.git", subdirectory = "t2v_metrics", branch = "main" }
fd-evaluator      = { git = "ssh://git@github.com/End2End-Diffusion/diffusionbench-evals.git", subdirectory = "fd_evaluator", branch = "main" }
```

Or pip:

```bash
pip install "git+https://github.com/End2End-Diffusion/diffusionbench-evals.git#subdirectory=dpgbench"
pip install "git+https://github.com/End2End-Diffusion/diffusionbench-evals.git#subdirectory=geneval"
pip install "git+https://github.com/End2End-Diffusion/diffusionbench-evals.git#subdirectory=t2v_metrics"
pip install "git+https://github.com/End2End-Diffusion/diffusionbench-evals.git#subdirectory=fd_evaluator"
```

Usage examples: see each subdir's README.

## Citation

Please cite the original benchmarks (linked above) and this repo:

```bibtex
@misc{diffusionbench-evals,
  title  = {diffusionbench-evals: A Unified codebase for ImageNet and text-to-image evaluation},
  author = {Leng, Xingjian and Singh, Jaskirat},
  year   = {2026},
  url    = {https://github.com/End2End-Diffusion/diffusionbench-evals}
}
```
