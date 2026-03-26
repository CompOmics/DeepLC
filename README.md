<img src="https://github.com/compomics/DeepLC/raw/master/img/deeplc_logo.png" width="150" height="150" /> <br/><br/>

[![GitHub release](https://flat.badgen.net/github/release/compomics/deeplc)](https://github.com/compomics/DeepLC/releases/latest/)
[![PyPI](https://flat.badgen.net/pypi/v/deeplc)](https://pypi.org/project/deeplc/)
[![Conda](https://img.shields.io/conda/vn/bioconda/deeplc?style=flat-square)](https://bioconda.github.io/recipes/deeplc/README.html)
[![GitHub Workflow Status](https://flat.badgen.net/github/checks/compomics/deeplc/)](https://github.com/compomics/deeplc/actions/)
[![License](https://flat.badgen.net/github/license/compomics/deeplc)](https://www.apache.org/licenses/LICENSE-2.0)


DeepLC: Retention time prediction for peptides carrying any modification.

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
---

## Introduction

DeepLC is a retention time predictor for peptides. Its strength lies in the fact that it can accurately predict retention times for modified peptides, even if hasn't seen said modification during training.

DeepLC can be used through the [web application](https://iomics.ugent.be/deeplc/) or as a Python package. In the latter case, DeepLC can be used from the command line, or as a Python module.

## Citation

If you use DeepLC for your research, please use the following citation:
>**DeepLC can predict retention times for peptides that carry as-yet unseen modifications**  
>Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens & Sven Degroeve  
> Nature Methods 18, 1363–1369 (2021) [doi: 10.1038/s41592-021-01301-5](http://dx.doi.org/10.1038/s41592-021-01301-5)

## Usage

### Web application
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iomics.ugent.be/deeplc/)

Just go to [iomics.ugent.be/deeplc](https://iomics.ugent.be/deeplc/) and get started!



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

DeepLC accepts any PSM file format supported by
[psm_utils](https://psm-utils.readthedocs.io/en/stable/api/psm_utils.io.html),
including MaxQuant msms.txt, Sage, MSAmanda, Percolator, and many more. The file
format is automatically inferred from the file extension, or can be specified
explicitly with the `--psm-filetype` option.

At a minimum, a tab-separated file with a `peptidoform` and `spectrum_id` column
is accepted. Peptidoforms must be in
[ProForma 2.0](https://pubs.acs.org/doi/10.1021/acs.jproteome.1c00771) notation.
For calibration or fine-tuning, a `retention_time` column is also required.

For example:

```tsv
spectrum_id	peptidoform	retention_time
0	AAGPSLSHTSGGTQSK/2	12.16
1	AAINQK[Acetyl]LIETGER/2	34.10
2	AANDAGYFNDEM[Oxidation]APIEVK[Acetyl]TK/3	37.38
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

See the [FAQ](https://deeplc.readthedocs.io/en/latest/faq.html) in the documentation.
