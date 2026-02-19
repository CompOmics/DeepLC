# DeepLC 4.0 to do list

## Alpha 1 release

[x] SciKit-Learn like API for calibration
[x] Streamlined use of Data and DataLoader
[x] Module with PyTorch-level model operations (train, predict, load, save)
[x] Refactor core functions to use new model operations module
[x] Add architecture module for training new models

## Alpha 2 release

[ ] Unit & integration tests
[ ] Retrain models with native pyTorch
[ ] Get calibration/finetuning PSMs from main psm_list using score/q-value for best selection? (Ralf)
[ ] Add CLI commands with file I/O
[ ] Integrate align.py functionality
[ ] Plot module: Update or move to MS²Rescore report module?

## Beta release

[ ] Ensure mapping of MaxQuant modifications
[ ] Update README
[ ] Update documentation to reflect new structure
[ ] Update examples to use new structure

## Stable release

[ ] Decent coverage of unit tests
[ ] Update GUI (no use of argparse -> alternative for Gooey?)
[ ] Update Streamlit app

## Open questions / issues

[ ] Should the library feature be reintroduced?
[ ] Implementation into IM2Deep
