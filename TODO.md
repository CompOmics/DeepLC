# DeepLC 4.0 to do list

## Alpha 1 release

[x] SciKit-Learn like API for calibration
[x] Streamlined use of Data and DataLoader
[x] Module with PyTorch-level model operations (train, predict, load, save)
[x] Refactor core functions to use new model operations module

## Alpha 2 release

[ ] Add architecture module for training new models
[ ] Get calibration/finetuning PSMs from main psm_list using score/q-value for best selection?
[ ] Add CLI commands with file I/O

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
