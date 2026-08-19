# dispute-desk
A multi-agent payment-dispute/chargeback copilot built on synthetic data modelled after real Razorpay and Stripe dispute taxonomies

V0: A simple python file which takes amount and delivery confirmation as inputs to classify the dispute.

## Requirements

Python **3.11+**. This repo pins `3.11.9` in `.python-version` so pyenv will not use the global 2.7 install.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

After that, `python` and `pytest` in this directory are Python 3:

```bash
python disputedesk.py
pytest -v
```
