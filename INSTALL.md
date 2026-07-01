# Installation

## Requirements

- Python ≥ 3.10
- A machine with sufficient RAM for TabPFN (EFE-Tab) and a CUDA-capable GPU
  for Chronos-2 (EFE-Time). EFE-Tab can run on CPU.
- Access to an LLM provider. The bundled configs target AWS Bedrock
  (Claude); the framework also supports any OpenAI-compatible endpoint.

## Install the framework + Python dependencies

From the repository root:

```bash
pip install -e .                    # installs the bundled openevolve framework
pip install -r requirements.txt     # method-specific deps (TabPFN, LightGBM, Chronos, …)
```

`pip install -e .` registers an `openevolve-run` console script and makes the
`openevolve` package importable. The same evolution can also be launched via
`python openevolve-run.py …` without installation if you prefer.

## EFE-Tab — extra dependencies

Already covered by `requirements.txt`:
- `tabpfn` — scoring model
- `lightgbm`, `optuna` — depth-3 tree feedback with HPO
- `scikit-learn`, `pandas`, `numpy`

For TabArena / OpenML datasets, the `tabarena_adapter` will install / fetch
metadata on first use.

## EFE-Time — extra dependencies

Already covered by `requirements.txt`:
- `chronos-forecasting`, `torch` — fixed Chronos-2 forecaster
- `scipy`, `numpy`

For GIFT-Eval datasets you also need:
```bash
pip install "git+https://github.com/SalesforceAIResearch/gift-eval.git"
```
and the dataset files themselves:
```bash
huggingface-cli download Salesforce/GiftEval \
  --repo-type=dataset \
  --local-dir /path/to/gift_eval_data \
  --include "hospital/*"            # or any other dataset name
export GIFT_EVAL=/path/to/gift_eval_data
```

## LLM provider authentication

### AWS Bedrock (default in both configs)

Either:
```bash
export AWS_BEARER_TOKEN_BEDROCK="your-ABSK-token"
```
or standard AWS credentials:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...        # optional
pip install boto3
```

### OpenAI-endpoint 

Edit the `llm:` section of the config:
```yaml
llm:
  provider: openai
  primary_model: "your-model"
  api_base: "https://your-endpoint/v1"
  api_key: "${OPENAI_API_KEY}"
```
Then `export OPENAI_API_KEY=...`.

## Verifying the install

```bash
python -c "import openevolve; from openevolve.evaluation_result import EvaluationResult; print('framework OK')"

# EFE-Tab module imports:
PYTHONPATH=EFE-Tab python -c "import evaluator, _evaluator_core, helpers, tabarena_adapter; print('EFE-Tab OK')"

# EFE-Time module imports:
PYTHONPATH=EFE-Time python -c "import evaluator, gift_eval_adapter; print('EFE-Time OK')"
```
