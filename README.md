<img src="https://github.com/compomics/DeepLC/raw/main/img/deeplc_logo.png" width="150" height="150" /> <br/><br/>

[![GitHub release](https://flat.badgen.net/github/release/compomics/deeplc)](https://github.com/compomics/DeepLC/releases/latest/)
[![PyPI](https://flat.badgen.net/pypi/v/deeplc)](https://pypi.org/project/deeplc/)
[![Conda](https://img.shields.io/conda/vn/bioconda/deeplc?style=flat-square)](https://bioconda.github.io/recipes/deeplc/README.html)
[![GitHub Workflow Status](https://flat.badgen.net/github/checks/compomics/deeplc/)](https://github.com/compomics/deeplc/actions/)
[![License](https://flat.badgen.net/github/license/compomics/deeplc)](https://www.apache.org/licenses/LICENSE-2.0)

---

**DeepLC: Retention time prediction for peptides carrying any modification.**

---

## About DeepLC

DeepLC predicts retention times for peptides carrying any modification. It does this by leveraging 
a deep learning model based on atomic composition features. Starting with v4, DeepLC comes with a
multitask pretrained model covering multiple LC setups, enabling accurate predictions out of the
box. For best results on a specific dataset, predictions can be calibrated or fine-tuned
using a small reference set of identified PSMs.

## Citation

If you use DeepLC, please cite:

> **DeepLC can predict retention times for peptides that carry as-yet unseen modifications**  
> Robbin Bouwmeester, Ralf Gabriels, Niels Hulstaert, Lennart Martens & Sven Degroeve  
> *Nature Methods* 18, 1363–1369 (2021) [doi:10.1038/s41592-021-01301-5](https://doi.org/10.1038/s41592-021-01301-5)

If you use the transfer learning functionality, please also cite:

> **Retention time prediction improves proteomics database search and identification rates**  
> *Nature Communications* (2026) [doi:10.1038/s41467-026-68981-5](https://doi.org/10.1038/s41467-026-68981-5)

To replicate the results from this paper, use DeepLC
[v3.1.13](https://github.com/compomics/deeplc/releases/v3.1.13). For regular use, we recommend the 
[latest stable version](https://github.com/compomics/deeplc/releases/latest).

## Usage

### Web application

A hosted web application is available at
[iomics.ugent.be/deeplc](https://iomics.ugent.be/deeplc/) — no installation required.

### Local graphical interface

**Windows:** download the one-click installer from the
[releases page](https://github.com/compomics/DeepLC/releases/latest).

**Other platforms:** install with GUI dependencies and launch as a desktop app or local web server:

```sh
pip install deeplc[gui]
deeplc gui            # opens in browser
deeplc gui --native   # opens as desktop window
```

### Command line and Python API

```sh
pip install deeplc
deeplc predict peptides.tsv
```

```python
from psm_utils.io import read_file
from deeplc import predict_and_calibrate

psm_list = read_file("peptides.tsv")
calibrated_rt = predict_and_calibrate(psm_list)
```

See the [documentation](https://deeplc.readthedocs.io) for the full CLI reference,
Python API, and input file format.

## Related projects

- [im2deep](https://github.com/compomics/im2deep) — Ion mobility / collisional cross section
  prediction using the same atomic composition approach
- [MS²Rescore](https://github.com/compomics/ms2rescore) — Peptide identification rescoring
  that uses DeepLC retention time predictions as a rescoring feature
- [iDeepLC](https://doi.org/10.1021/acs.analchem.5c08017) — DeepLC variant using molecular
  descriptors to incorporate full molecular structure into predictions; better performance for
  some amino acid modifications.

## Documentation

Full documentation at [deeplc.readthedocs.io](https://deeplc.readthedocs.io):

- [Usage](https://deeplc.readthedocs.io/en/latest/usage/) — CLI, Python API, input formats
- [Prediction models](https://deeplc.readthedocs.io/en/latest/models/) — Model descriptions and training data
- [Migrating from v3](https://deeplc.readthedocs.io/en/latest/migration/) — API and format changes
- [Changelog](https://deeplc.readthedocs.io/en/latest/changelog/)
- [Contributing](https://deeplc.readthedocs.io/en/latest/contributing/)
