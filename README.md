# **PROJECT IS STILL IN PROGRESS** 
# Fourier-Transform Decoding of Periodic Neural Dynamics (Project Page)

This repository showcases my summer research project analyzing zebrafish calcium imaging with a Fourier-based pipeline to detect stimulus-locked neurons under periodic perturbations.

**Live Project Page:** (will appear after enabling GitHub Pages)  
**Manuscript PDF:** See the project page or `docs/assets/manuscript.pdf`

## Highlights
- Fourier pipeline extracts amplitude & phase at the known stimulus frequency.
- Captures transient onset/offset responses that regression may miss.
- Reproduces/analyzes a public zebrafish dataset; compares with regression-based labeling.

## Repo Structure
- `docs/` — the GitHub Pages site (edit `docs/index.md` to change the page)
- `demo/` — analysis code (Python)
- `notebooks/` — Jupyter notebooks (exploratory / figures)
- `figures/` — exported plots used on the page
- `data/` — small samples or instructions to download data

## Quickstart
1. Create a Python env and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the example script to compute Fourier components for a toy trace:
   ```bash
   python demo/example_fft.py
   ```
3. Open `notebooks/01_demo.ipynb` to see amplitude/phase extraction workflow.

## Citation
If you use or discuss this work, please cite the repository.

## License
- Code: MIT (see `LICENSE`).
- Text/figures: CC BY 4.0 (update if you prefer).
