This repository contains the code that was used to evaluate and compare deterministic watson samples in the paper "Deterministic Sampling of the Watson Distribution" for the FUSION26 conference.

run with

```bash
uv run evaluation.py

# or 
uv run evaluation.py --silent -n 10000000
```

Minimal sampling helper:

```python
from sample import sample

# S2 example
points = sample(kappa=5.0, dim=2, sample_count=99, method="sobol")
```

Supported methods:
- S2 (`dim=2`): `kronecker`, `random`, `sobol`, `fibonacci-rank1` (requires `sample_count + 1` to be Fibonacci).
- S3 (`dim=3`): `kronecker` (up to n=177), `random`, `sobol`, `rank1` (up to n=155), `rank1_cbc` (only prime n).

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/1d0e30f9-f558-4910-9aa3-a41c23635192" />
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/71e0b6a2-fdd5-4563-be56-a0f33f503dd6" />
