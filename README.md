# CLAIM Implementation

This repository contains the code for the CLAIM (Causally-Learned Adaptive and Iterative Mechanism) module, which extends the [AIM mechanism](https://github.com/ryan112358/mbi).

## Repository Structure

To ensure a clean separation between the original library and our new contributions, the repository is split into two main directories:

- `AIM_Implementation/`: Contains the original, unmodified AIM library. This is based exactly on commit `025b76f84f40256529732e98d93c1139fb2153d8` from the `ryan112358/mbi` repository. It includes performance and memory optimizations not present in earlier versions.
- `CLAIM_Implementation/`: Contains the modifications and new mechanisms for CLAIM (`claim.py` and `fwl.py`).

## Environment Setup

1. Create a new virtual environment:
   ```bash
   python -m venv aim_env
   source aim_env/bin/activate
   ```
2. Install the required dependencies (frozen from the original CLAIM environment):
   ```bash
   pip install -r requirements.txt
   ```
3. Install the core `mbi` library in editable mode so it can be imported globally by CLAIM:
   ```bash
   cd AIM_Implementation
   pip install -e .
   ```

## Running the Code

You can run the code directly from the `CLAIM_Implementation` directory:

```bash
cd CLAIM_Implementation
python claim.py --dataset adult
```
