# Agentics Builder

## Environment

- requires-python = "3.11.12"

```
conda create --name agentics python=3.11.12

conda activate agentics
```

## Setup

```
pip install -r requirements.txt
```

- Update `.env`
  - Add OPENROUTER_API_KEY (Always Required. Request from https://openrouter.ai/settings/keys)
  - Add other keys (Opinioal based on models in basic.yaml. Get key name from https://aider.chat/docs/llms.html)


## Config

### Specs

- Update specs in `specs/spec.md`
  - {highleveltasks}
  - {lowleveltasks}

- Update model in `specs/basic.yaml`

### Build Agent

```
python director.py --config specs/basic.yaml
```

### Run Agent

```
python agent.py
```

---

Created with 🤖 by [agentics.world](https://agentics.world)