# Evolutionary Feature Engineering (EFE)

Code for the paper **Evolutionary Feature Engineering for Structured Data**.

EFE uses LLM-guided evolutionary search to discover feature-engineering programs for structured data. The downstream model stays fixed. EFE only changes the preprocessing transformation placed before it.

- **EFE-Time** evolves invertible normalizations for time-series forecasting.
- **EFE-Tab** evolves compact feature generator programs for tabular classification.

The includes the EFE pipelines and the runners for evolving, validating, and scoring candidate programs.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -r requirements.txt
```

The default configs use AWS Bedrock. Set your LLM credentials before running:

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-token"
```

OpenAI-compatible endpoints can also be configured in the relevant `config.yaml` file.

## Quick Start

### EFE-Time

```bash
export GIFT_EVAL=/path/to/gift_eval_data
export TS_DATASET=covid_deaths

python openevolve-run.py \
  EFE-Time/initial_program.py \
  EFE-Time/evaluator.py \
  --config EFE-Time/config.yaml \
  --iterations 100 \
  --output EFE-Time/openevolve_output
```

### EFE-Tab

```bash
export TABARENA_TASK=churn

python openevolve-run.py \
  EFE-Tab/initial_program.py \
  EFE-Tab/evaluator.py \
  --config EFE-Tab/config.yaml \
  --iterations 100 \
  --output EFE-Tab/openevolve_output
```

Outputs are written to the selected `openevolve_output/` directory.

## Repository Structure

```text
EFE/
├── EFE-Time/          # time-series transformation evolution
├── EFE-Tab/           # tabular feature-program evolution
├── openevolve/        # bundled evolutionary program-search framework
├── openevolve-run.py  # command-line entry point
├── requirements.txt
└── INSTALL.md
```

Method-specific setup details are available in `EFE-Time/README.md`, `EFE-Tab/README.md`, and `INSTALL.md`.

## Citation

If you use this repository, please cite:

```bibtex
@misc{taga2026evolutionaryfeatureengineeringstructured,
      title={Evolutionary Feature Engineering for Structured Data}, 
      author={Ege Onur Taga and Yilin Zhuang and M. Emrullah Ildiz and Petros Mol and Abhimanyu Das and Karthik Duraisamy and Samet Oymak},
      year={2026},
      eprint={2607.01548},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.01548}, 
}
```

## Acknowledgments

This repository builds on [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) as its evolutionary program-search framework.

## License

This project is released under the Apache-2.0 license.
