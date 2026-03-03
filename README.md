<img src="https://github.com/compomics/DeepLC/raw/master/img/deeplc_logo.png" width="150" height="150" /> <br/><br/>

[![GitHub release](https://flat.badgen.net/github/release/compomics/deeplc)](https://github.com/compomics/DeepLC/releases/latest/)
[![PyPI](https://flat.badgen.net/pypi/v/deeplc)](https://pypi.org/project/deeplc/)
[![Conda](https://img.shields.io/conda/vn/bioconda/deeplc?style=flat-square)](https://bioconda.github.io/recipes/deeplc/README.html)
[![GitHub Workflow Status](https://flat.badgen.net/github/checks/compomics/deeplc/)](https://github.com/compomics/deeplc/actions/)
[![License](https://flat.badgen.net/github/license/compomics/deeplc)](https://www.apache.org/licenses/LICENSE-2.0)
[![Twitter](https://flat.badgen.net/twitter/follow/compomics?icon=twitter)](https://twitter.com/compomics)

DeepLC: Retention time prediction for (modified) peptides using Deep Learning.

---

- [Introduction](#introduction)
- [Citation](#citation)
- [Usage](#usage)
  - [Web application](#web-application)
  - [Graphical user interface](#graphical-user-interface)
  - [Python package](#python-package)
    - [Installation](#installation)
    - [Command line interface](#command-line-interface)
    - [Python module](#python-module)
  - [Input files](#input-files)
  - [Prediction models](#prediction-models)
- [Q&A](#qa)

---

## Introduction

DeepLC is a retention time predictor for (modified) peptides that employs Deep
Learning. Its strength lies in the fact that it can accurately predict
retention times for modified peptides, even if hasn't seen said modification
during training.

DeepLC can be used through the
[web application](https://iomics.ugent.be/deeplc/),
locally with a graphical user interface (GUI), or as a Python package. In the
latter case, DeepLC can be used from the command line, or as a Python module.

## Citation

If you use DeepLC for your research, please use the following citation:
>**DeepLC can predict retention times for peptides that carry as-yet unseen modifications**  
>Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens & Sven Degroeve  
> Nature Methods 18, 1363–1369 (2021) [doi: 10.1038/s41592-021-01301-5](http://dx.doi.org/10.1038/s41592-021-01301-5)

## Usage

### Web application
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iomics.ugent.be/deeplc/)

Just go to [iomics.ugent.be/deeplc](https://iomics.ugent.be/deeplc/) and get started!


### Graphical user interface

#### In an existing Python environment (cross-platform)

1. In your terminal with Python (>=3.7) installed, run `pip install deeplc[gui]`
2. Start the GUI with the command `deeplc-gui` or `python -m deeplc.gui`

#### Standalone installer (Windows)

[![Download GUI](https://flat.badgen.net/badge/download/GUI/blue)](https://github.com/compomics/DeepLC/releases/latest/)


1. Download the DeepLC installer (`DeepLC-...-Windows-64bit.exe`) from the
[latest release](https://github.com/compomics/DeepLC/releases/latest/)
2. Execute the installer
3. If Windows Smartscreen shows a popup window with "Windows protected your PC",
click on "More info" and then on "Run anyway". You will have to trust us that
DeepLC does not contain any viruses, or you can check the source code 😉
4. Go through the installation steps
5. Start DeepLC!

![GUI screenshot](https://github.com/compomics/DeepLC/raw/master/img/gui-screenshot.png)


### Python package

#### Installation

[![install with bioconda](https://flat.badgen.net/badge/install%20with/bioconda/green)](http://bioconda.github.io/recipes/deeplc/README.html)
[![install with pip](https://flat.badgen.net/badge/install%20with/pip/green)](http://bioconda.github.io/recipes/deeplc/README.html)
[![container](https://flat.badgen.net/badge/pull/biocontainer/green)](https://quay.io/repository/biocontainers/deeplc)

Install with conda, using the bioconda and conda-forge channels:
`conda install -c bioconda -c conda-forge deeplc`

Or install with pip:
`pip install deeplc`

#### Command line interface

To use the DeepLC CLI, run:

```sh
deeplc --file_pred <path/to/peptide_file.csv>
```

We highly recommend to add a peptide file with known retention times for
calibration:

```sh
deeplc --file_pred  <path/to/peptide_file.csv> --file_cal <path/to/peptide_file_with_tr.csv>
```

For an overview of all CLI arguments, run `deeplc --help`.

#### Python module

Minimal example:

```python
import pandas as pd
from deeplc import DeepLC

peptide_file = "datasets/test_pred.csv"
calibration_file = "datasets/test_train.csv"

pep_df = pd.read_csv(peptide_file, sep=",")
pep_df['modifications'] = pep_df['modifications'].fillna("")

cal_df = pd.read_csv(calibration_file, sep=",")
cal_df['modifications'] = cal_df['modifications'].fillna("")

dlc = DeepLC()
dlc.calibrate_preds(seq_df=cal_df)
preds = dlc.make_preds(seq_df=pep_df)
```

Minimal example with psm_utils:

```python
import pandas as pd

from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from psm_utils.io import write_file

from deeplc import DeepLC

infile = pd.read_csv("https://github.com/compomics/DeepLC/files/13298024/231108_DeepLC_input-peptides.csv")
psm_list = []

for idx,row in infile.iterrows():
    seq = row["modifications"].replace("(","[").replace(")","]")
    
    if seq.startswith("["):
        idx_nterm = seq.index("]")
        seq = seq[:idx_nterm+1]+"-"+seq[idx_nterm+1:]
        
    psm_list.append(PSM(peptidoform=seq,spectrum_id=idx))

psm_list = PSMList(psm_list=psm_list)

infile = pd.read_csv("https://github.com/compomics/DeepLC/files/13298022/231108_DeepLC_input-calibration-file.csv")
psm_list_calib = []

for idx,row in infile.iterrows():
    seq = row["seq"].replace("(","[").replace(")","]")
    
    if seq.startswith("["):
        idx_nterm = seq.index("]")
        seq = seq[:idx_nterm+1]+"-"+seq[idx_nterm+1:]
        
    psm_list_calib.append(PSM(peptidoform=seq,retention_time=row["tr"],spectrum_id=idx))

psm_list_calib = PSMList(psm_list=psm_list_calib)

dlc = DeepLC()
dlc.calibrate_preds(psm_list_calib)
preds = dlc.make_preds(seq_df=psm_list)
```

For a more elaborate example, see
[examples/deeplc_example.py](https://github.com/compomics/DeepLC/blob/master/examples/deeplc_example.py)
.

### Input files

DeepLC expects comma-separated values (CSV) with the following columns:

- `seq`: unmodified peptide sequences
- `modifications`: MS2PIP-style formatted modifications: Every modification is
  listed as `location|name`, separated by a pipe (`|`) between the location, the
  name, and other modifications. `location` is an integer counted starting at 1
  for the first AA. 0 is reserved for N-terminal modifications, -1 for
  C-terminal modifications. `name` has to correspond to a Unimod (PSI-MS) name.
- `tr`: retention time (only required for calibration)

For example:

```csv
seq,modifications,tr
AAGPSLSHTSGGTQSK,,12.1645
AAINQKLIETGER,6|Acetyl,34.095
AANDAGYFNDEMAPIEVKTK,12|Oxidation|18|Acetyl,37.3765
```

See
[examples/datasets](https://github.com/compomics/DeepLC/tree/master/examples/datasets)
for more examples.

### Prediction models

DeepLC comes with multiple CNN models trained on data from various experimental
settings. By default, DeepLC selects the best model based on the calibration dataset. If
no calibration is performed, the first default model is selected. Always keep
note of the used models and the DeepLC version. The current version comes with:

| Model filename | Experimental settings | Publication |
| - | - | - |
| full_hc_PXD005573_mcp_8c22d89667368f2f02ad996469ba157e.hdf5 | Reverse phase | [Bruderer et al. 2017](https://pubmed.ncbi.nlm.nih.gov/29070702/) |
| full_hc_PXD005573_mcp_1fd8363d9af9dcad3be7553c39396960.hdf5 | Reverse phase | [Bruderer et al. 2017](https://pubmed.ncbi.nlm.nih.gov/29070702/) |
| full_hc_PXD005573_mcp_cb975cfdd4105f97efa0b3afffe075cc.hdf5 | Reverse phase | [Bruderer et al. 2017](https://pubmed.ncbi.nlm.nih.gov/29070702/) |

For all the full models that can be used in DeepLC (including some TMT models!) please see:

[https://github.com/RobbinBouwmeester/DeepLCModels](https://github.com/RobbinBouwmeester/DeepLCModels)

Naming convention for the models is as follows:

[full_hc]\_[dataset]\_[fixed_mods]\_[hash].hdf5

The different parts refer to:

**full_hc** - flag to indicated a finished, trained, and fully optimized model

**dataset** - name of the dataset used to fit the model (see the original publication, supplementary table 2)

**fixed mods** - flag to indicate fixed modifications were added to peptides without explicit indication (e.g., carbamidomethyl of cysteine)

**hash** - indicates different architectures, where "1fd8363d9af9dcad3be7553c39396960" indicates CNN filter lengths of 8, "cb975cfdd4105f97efa0b3afffe075cc" indicates CNN filter lengths of 4, and "8c22d89667368f2f02ad996469ba157e" indicates filter lengths of 2


## Q&A

**__Q: Is it required to indicate fixed modifications in the input file?__**

Yes, even modifications like carbamidomethyl should be in the input file.

**__Q: So DeepLC is able to predict the retention time for any modification?__**

Yes, DeepLC can predict the retention time of any modification. However, if the
modification is **very** different from the peptides the model has seen during
training the accuracy might not be satisfactory for you. For example, if the model
has never seen a phosphor atom before, the accuracy of the prediction is going to
be low.

**__Q: Installation fails. Why?__**

Please make sure to install DeepLC in a path that does not contain spaces. Run
the latest LTS version of Ubuntu or Windows 10. Make sure you have enough disk
space available, surprisingly TensorFlow needs quite a bit of disk space. If
you are still not able to install DeepLC, please feel free to contact us:

Robbin.Bouwmeester@ugent.be and Ralf.Gabriels@ugent.be

**__Q: I have a special usecase that is not supported. Can you help?__**

Ofcourse, please feel free to contact us:

Robbin.Bouwmeester@ugent.be and Ralf.Gabriels@ugent.be

**__Q: DeepLC runs out of memory. What can I do?__**

You can try to reduce the batch size. DeepLC should be able to run if the batch size is low
enough, even on machines with only 4 GB of RAM.

**__Q: I have a graphics card, but DeepLC is not using the GPU. Why?__**

For now DeepLC defaults to the CPU instead of the GPU. Clearly, because you want
to use the GPU, you are a power user :-). If you want to make the most of that expensive
GPU, you need to change or remove the following line (at the top) in __deeplc.py__:

```
# Set to force CPU calculations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

Also change the same line in the function __reset_keras()__:

```
# Set to force CPU calculations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

Either remove the line or change to (where the number indicates the number of GPUs):

```
# Set to force CPU calculations
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
```

**__Q: What modification name should I use?__**

The names from unimod are used. The PSI-MS name is used by default, but the Interim name
is used as a fall-back if the PSI-MS name is not available. It should be fine as long as it is support by [proforma](https://pubs.acs.org/doi/10.1021/acs.jproteome.1c00771) and [psm_utils](https://github.com/compomics/psm_utils).

**__Q: I have a modification that is not in unimod. How can I add the modification?__**

Unfortunately since the V3.0 this is not possible any more via the GUI or commandline. You will need to use [psm_utils](https://github.com/compomics/psm_utils), above a minimal example is shown where we convert an identification file into a psm_list which is accepted by DeepLC. Here the sequence can for example include just the composition in proforma format (e.g., SEQUEN[Formula:C12H20O2]CE).

**__Q: Help, all my predictions are between [0,10]. Why?__**

It is likely you did not use calibration. No problem, but the retention times for training
purposes were normalized between [0,10]. This means that you probably need to adjust the
retention time yourselve after analysis or use a calibration set as the input.


**__Q: What does the option `dict_divider` do?__**

This parameter defines the precision to use for fast-lookup of retention times
for calibration. A value of 10 means a precision of 0.1 (and 100 a precision of
0.01) between the calibration anchor points. This parameter does not influence
the precision of the calibration, but setting it too high might mean that there
is bad selection of the models between anchor points. A safe value is usually
higher than 10.


**__Q: What does the option `split_cal` do?__**

The option `split_cal`, or split calibration, sets number of divisions of the
chromatogram for piecewise linear calibration. If the value is set to 10 the
chromatogram is split up into 10 equidistant parts. For each part the median
value of the calibration peptides is selected. These are the anchor points.
Between each anchor point a linear fit is made. This option has no effect when
the pyGAM generalized additive models are used for calibration.


**__Q: How does the ensemble part of DeepLC work?__**

Models within the same directory are grouped if they overlap in their name. The overlap
has to be in their full name, except for the last part of the name after a "_"-character.

The following models will be grouped:

```
full_hc_dia_fixed_mods_a.hdf5
full_hc_dia_fixed_mods_b.hdf5
```

None of the following models will not be grouped:

```
full_hc_dia_fixed_mods2_a.hdf5
full_hc_dia_fixed_mods_b.hdf5
full_hc_dia_fixed_mods_2_b.hdf5
```

**__Q: I would like to take the ensemble average of multiple models, even if they are trained on different datasets. How can I do this?__**

Feel free to experiment! Models within the same directory are grouped if they overlap in
their name. The overlap has to be in their full name, except for the last part of the
name after a "_"-character.

The following models will be grouped:

```
model_dataset1.hdf5
model_dataset2.hdf5
```

So you just need to rename your models.
