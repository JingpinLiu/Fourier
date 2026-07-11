# Fourier-Transform Decoding of Periodic Neural Dynamics

This repository contains the full analysis pipeline for a research project analyzing zebrafish calcium imaging with a Fourier-based approach to detect stimulus-locked neurons under periodic perturbations (phototaxis). It reproduces every figure in the accompanying manuscript, end to end, from a publicly available dataset.

This approach is conceptually related to the **steady-state response (SSR)** paradigm long used in MEG/EEG research (e.g., steady-state visual/auditory evoked potentials, SSVEP/SSAEP): both drive the system with a periodic stimulus and extract the amplitude and phase of the response at the stimulation frequency via Fourier analysis, rather than relying on trial-averaged time-domain waveforms. Here, the same idea is applied at single-neuron resolution in calcium imaging data, rather than at the sensor level in M/EEG.

**Manuscript PDF:** [`docs/assets/Manuscript v0.3.pdf`](docs/assets/Manuscript%20v0.3.pdf) (also linked from the [project page](https://jingpinliu.github.io/Fourier/))

## Quick Start
1. Set up the environment: create a virtual environment and `pip install -r requirements.txt` (see the setup cell at the top of `notebooks/Fourier_Analysis_Demo.ipynb` for the exact commands).
2. New to Fourier transforms? Start with [`notebooks/Intro_Fourier_Transformation.ipynb`](notebooks/Intro_Fourier_Transformation.ipynb) — it builds the DFT concepts (frequency, amplitude, phase) from scratch using toy signals.
3. Work through [`notebooks/Fourier_Analysis_Demo.ipynb`](notebooks/Fourier_Analysis_Demo.ipynb) cell by cell to reproduce the full pipeline — from raw stimulus/neuron traces to the spatial maps and regression-validation figures in the manuscript — on one example zebrafish (Fish 6), whose data is included in `data/` (the large per-neuron activity file is fetched automatically on first run).
   - Prefer not to run a notebook cell by cell? [`demo/run_fish6_pipeline.py`](demo/run_fish6_pipeline.py) runs the identical pipeline start to finish as a plain script — `python demo/run_fish6_pipeline.py` — and saves every figure (named after its manuscript figure number) plus a `summary.json` of key metrics to `results/fish6/`, with no Jupyter required.
4. Want to reproduce the multi-fish results (Figures 4 & 10, averaged across all 10 fish)? See [`demo/multi_fish_accuracy.py`](demo/multi_fish_accuracy.py) — this requires downloading each fish's data separately (see the script's docstring).

## Highlights
- Fourier pipeline extracts amplitude & phase at the known stimulus frequency to identify stimulus-locked neurons.
- Captures transient onset/offset responses that regression-based labeling may miss.
- Reproduces the manuscript's analysis of a public zebrafish dataset, including spatial/phase brain maps and quantitative validation against regression-based labeling.

## Repo Structure
- `docs/` — the GitHub Pages project site and the manuscript PDF
- `demo/` — standalone analysis scripts: `example_fft.py` (minimal toy FFT demo), `run_fish6_pipeline.py` (headless, script version of the full single-fish notebook pipeline), `multi_fish_accuracy.py` (multi-fish extension of the pipeline)
- `demo_functions.py` — shared Fourier-analysis helper functions used throughout the notebooks
- `notebooks/` — the two Jupyter notebooks described above
- `images/` — images used in the notebooks and project page
- `data/` — the example fish's small metadata/coordinate files, plus instructions for the full dataset (see `data/README.md`)

## Citation
If you use or discuss this work, please cite the repository and manuscript:

```
Liu, J. D. Fourier-Transform Decoding of Periodic Neural Dynamics in Zebrafish Calcium Imaging.
GitHub repository: https://github.com/JingpinLiu/Fourier
```

## License
- Code: MIT (see `LICENSE`).
- Text/figures: CC BY 4.0.
