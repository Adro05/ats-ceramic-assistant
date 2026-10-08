# ats-ceramic-assistant

Prototype of a batch shade calibration assistant for digitally printed ceramic tiles.

> **Status: foundation phase only (interim README).** All data in this repository is
> **synthetic**. No client data, printer details, instrument settings or tolerances are
> used or implied. Every threshold in `configs/` is a labelled placeholder. A full README
> (architecture, limitations, what is and is not implemented) will replace this file in a
> later chunk.

## Install and test

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```