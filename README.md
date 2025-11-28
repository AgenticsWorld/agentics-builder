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
  - Update with your keys


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