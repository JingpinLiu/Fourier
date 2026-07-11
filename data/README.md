**Notebook for Example Data**

- The data used for this project comes from a public neuron-imaging dataset for zebrafish. This dataset was used in the paper titled "Brain-wide Organization of Neuronal Activity and Convergent Sensorimotor Transformations in Larval Zebrafish<sup>1</sup>" [(Neuron 2018)](https://www.cell.com/neuron/pdf/S0896-6273%2818%2930844-4.pdf). The data consists of recordings of whole-brain recordings from larval zebrafish, fictively behaving in a virtual environment, imaged using a light-sheet microscope. The data is available for download on [FigShare](https://doi.org/10.25378/janelia.7272617).

1. Brain-wide Organization of Neuronal Activity and Convergent Sensorimotor Transformations in Larval Zebrafish. Xiuye Chen\*, Yu Mu\*, Yu Hu\*, Aaron T. Kuan\*, Maxim Nikitchenko, Owen Randlett, Alex B. Chen, Jeffery P. Gavornik, Haim Sompolinsky, Florian Engert, and Misha B. Ahrens (\*: equal contributions). Neuron, 2018. DOI: <https://doi.org/10.1016/j.neuron.2018.09.042>

## What's in this folder

All files here are for **Fish 6**, the example fish used throughout `notebooks/Fourier_Analysis_Demo.ipynb` (and one of the 10 fish analyzed in the manuscript):

| File | Contents |
|---|---|
| `stimulus.xlsx`, `timelist.xlsx` | Per-frame stimulus code and trial-order frame reindexing |
| `CellXYZ.xlsx`, `CellXYZnorm.xlsx`, `absIX.xlsx` | Neuron coordinates (raw and normalized to a common brain template) and their index into the activity matrix |
| `outline_xy.xlsx`, `outline_yz.xlsx`, `outline_zx.xlsx` | Brain outline coordinates (common template) used for the top/side/front spatial map projections |
| `behavior_full.xlsx`, `motorseed.xlsx` | Behavioral/motor recordings (not used by the Fourier pipeline itself, included for completeness) |
| `data_full.mat` | The unfiltered source metadata this fish's exports were derived from (coordinates, stimulus, behavior, anatomy stack, and the invalid-anatomy neuron indices used to filter down to the exported neuron count) |

**Not included** (downloaded automatically): `CellRespZ.h5`, the ~1.3 GB neuron x frame activity matrix (92,538 neurons x 3,780 frames) for Fish 6. The setup cell in `Fourier_Analysis_Demo.ipynb` downloads this automatically on first run.

## Reproducing the multi-fish results (Figures 4 & 10)

The manuscript averages its validation results across all 10 fish (IDs 6, 7, 10, 12-18). To reproduce this with `demo/multi_fish_accuracy.py`, download each fish's `TimeSeries.h5`, `CellXYZnorm.xlsx`, `stimulus.xlsx`, and `timelist.xlsx` from the [FigShare dataset](https://doi.org/10.25378/janelia.7272617) into `subject_<ID>/` subfolders — see that script's docstring for the exact expected layout.
