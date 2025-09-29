---
layout: default
title: "Fourier-Transform Decoding of Periodic Neural Dynamics"
---

# Fourier-Transform Decoding of Periodic Neural Dynamics in Zebrafish Calcium Imaging

_Preprint / project page — work in progress._

- **Manuscript (PDF):** [Download](assets/manuscript.pdf)
- **Code:** Browse [`src/`](../src/) and [`notebooks/`](../notebooks/)
- **Figures:** See [`figures/`](../figures/)

## TL;DR
We propose a Fourier-based pipeline to identify stimulus-locked neurons under periodic perturbations in zebrafish calcium imaging.

## What’s here
- **Concept:** Extract amplitude (response strength) and phase (response timing) at the known stimulus frequency.
- **Why it helps:** Captures transient onset/offset responses that standard regression may miss.
- **Validation:** Agreement with regression-based labels, with complementary coverage of phase-shifted neurons.

## Key Results (teasers)
<p>
<img src="../figures/toy_signal.png" alt="toy signal" width="45%">
<img src="../figures/amp_spectrum.png" alt="amp spectrum" width="45%">
</p>

## Reproduce
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/example_fft.py
```

## Manuscript
The full write-up (methods, figures, validations) is in the PDF above.

## Contact
your.email@example.com
