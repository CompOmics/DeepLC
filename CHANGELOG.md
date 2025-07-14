# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.9] - 2025-07-14

### Changed

- When no calibration peptides are present, just fit a model that returns the original predicted value

## [3.1.8] - 2025-02-21

### Changed

- Allow for much smaller peptides in feature calculation

## [3.1.7] - 2025-02-03

### Changed

- Log warnings in obtaining atoms only once

## [3.1.6] - 2025-02-03

### Changed

- Remove unimod, replace with psm_utils

## [3.1.5] - 2025-02-03

### Changed

- Bump minimal requirements python version

## [3.1.4] - 2025-02-03

### Changed

- Bump minimal requirements python version

## [3.1.3] - 2024-11-22

### Changed

- Bioconda fix import


## [3.1.2] - 2024-11-21

### Changed

- Remove dependencies

## [3.1.1] - 2024-10-10

### Changed

- Revert to linear calibration after transfer learning only

## [3.1.0] - 2024-08-31

### Changed

- Use scikit-learn instead of pygam for calibration

## [3.0.8] - 2024-08-25

### Changed

- Fix custom activation by using default implementation
- Loosen requirements

## [3.0.7] - 2024-08-19

### Changed

- Remove batch number parameter from GUI

## [3.0.6] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.5] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.4] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.3] - 2024-08-08

### Changed

- Remove old python build

## [3.0.2] - 2024-08-08

### Changed

- Debugging release issues

## [3.0.1] - 2024-08-08

### Changed

- Windows release fix

## [3.0.0] - 2024-08-08

### Changed

- New TensorFlow versions that break support with earlier models

## [2.2.38] - 2024-07-01

### Changed

- Relax tensorflow version for mac

## [2.2.37] - 2024-07-01

### Changed

- Pin scipy version

## [2.2.36] - 2024-04-14

### Changed

- Set max threads

## [2.2.35] - 2024-04-14

### Changed

- Remove limit threads feature calc

## [2.2.34] - 2024-04-13

### Fixed

- Fix issue transfer learning single model mode

## [2.2.33] - 2024-04-13

### Changed

- Make single model mode available and the default

## [2.2.32] - 2024-02-16

### Changed

- Bump version

## [2.2.31] - 2024-02-16

### Changed

- Bump version

## [2.2.30] - 2024-02-16

### Changed

- Bump version with new workflows

## [2.2.29] - 2024-02-16

### Changed

- Bump version with .toml

## [2.2.28] - 2024-02-16

### Changed

- Removed dependencies (most importantly sklearn; which is optional now)

## [2.2.27] - 2024-01-22

### Changed

- Support for custom labels in plot

## [2.2.26] - 2023-11-15

### Fixed

- Fix memory usage, limit threads

## [2.2.25] - 2023-11-15

### Fixed

- Fixed multiprocessing

## [2.2.24] - 2023-11-14

### Changed

- Pass flag CCS feature extract

## [2.2.23] - 2023-11-13

### Changed

- Reintroduce ability to predict CCS

## [2.2.22] - 2023-09-22

### Fixed

- Fix pypi token GA

## [2.2.21] - 2023-09-22

### Fixed

- Fix non-initialized modifications

### Fixed

- Allow for setting number of epochs

## [2.2.20] - 2023-09-19

### Fixed

- Fix selenium atom logging error

## [2.2.19] - 2023-09-19

### Fixed

- Fix selenium atom on AA

## [2.2.18] - 2023-09-18

### Fixed

- Fix wrong error except

## [2.2.17] - 2023-09-18

### Fixed

- Fix peptides that are too long and their modifcations

## [2.2.16] - 2023-09-18

### Changed

- bump version

## [2.2.15] - 2023-09-18

### Changed

- reduce logging (change to debug)

## [2.2.14] - 2023-09-03

### Fixed

- Fix missing debug time in feature extractor

## [2.2.13] - 2023-09-03

### Fixed

- Fix potential issues in feat extractor (uncommon AA)

## [2.2.12] - 2023-08-19

### Changed

- Set plotly param to false

## [2.2.11] - 2023-08-19

### Changed

- Remove assumption psm_utils_obj

## [2.2.10] - 2023-08-19

### Changed

- Add plotly diagnostic plots

## [2.2.9] - 2023-08-08

### Fixed

- Fix logger issue

## [2.2.8] - 2023-08-08

### Changed

- Activate garbage collection to clear GPU memory

## [2.2.7] - 2023-08-03

### Changed

- Reintroduce support for batched calculation
- Introduce batch_num_tf parameter for DeepLC that determines TF batch size

## [2.2.6] - 2023-08-02

### Changed

- Allow for linear piecewise calibration again
- More strict version of TF, as 2.13.0 crashes for transfer learning

## [2.2.5] - 2023-08-02

### Fixed

- Fixed a bug where a reinit led to issues with setting parallelism

## [2.2.4] - 2023-07-06

### Fixed

- Fixed a bug where it checked the calibration file while this was not initialized

## [2.2.3] - 2023-06-28

### Fixed

- Fixed a bug where missing atoms in encoding cause a crash

## [2.2.2] - 2023-06-23

### Fixed

- Fixed a bug where isotopes are incorrectly parsed

## [2.2.1] - 2023-06-18

### Fixed

- Fixed a bug where numpy no longer accepts dict_values, explicit list conversion

## [2.2.0] - 2023-06-14

### Fixed

- Fixed a bug where atom counts were wrong, fixed by retraining models with new features

## [2.1.9] - 2023-05-09

### Fixed

- Fixed a bug where detection of legacy csv is wrong

## [2.1.8] - 2023-05-09

### Changed

- remove pygam from GUI, set to true as default

## [2.1.7] - 2023-05-09

### Fixed

- fix setting cmd line calibration default

## [2.1.6] - 2023-05-09

### Fixed

- fix setting GUI calibration default

## [2.1.5] - 2023-05-08

### Fixed

- fix library feature

## [2.1.4] - 2023-05-08

### Changed

- slight refractoring

## [2.1.3] - 2023-05-08

### Changed

- slight refractoring

## [2.1.2] - 2023-05-08

### Changed

- add Arthur Declercq as contributor

## [2.1.2] - 2023-05-08

### Fixed

- dependency fix

## [2.1.1] - 2023-05-08

### Fixed

- Calibration bug fix

## [2.1.0] - 2023-05-08

### Changed

- Support for GUI and PSM utils
- Support for GUI and transfer learning

## [2.0.4] - 2023-05-06

### Changed

- Log peptide sequences that are too long

## [2.0.3] - 2023-05-02

### Changed

- bump version
- second try

## [2.0.2] - 2023-05-02

### Changed

- bump version

## [2.0.1] - 2023-05-02

### Changed

- bug fix for sequences > 60 AA
- bug fix non-resolved modifications

## [2.0.0] - 2023-04-30

### Added

- psm_utils integration
- pygam default calibrator

### Changed

- additional support transfer learning
- code refractoring of feature extraction

### Removed

- prediction library support
- piecewise linear calibration support

## [1.2.1] - 2023-02-13

### Changed

- Bump version

## [1.2.0] - 2023-02-13

### Added

- Support for DeepLCRetrainer

### Changed

- Optimized library usage (~85% reduce in DeepLC prediction time) @markmipt

## [1.1.2] - 2022-04-06

### Changed

- Made Gooey an optional dependency (facilitates install on Linux). Install the optional
  dependencies for the graphical user interface with `pip install deeplc[gui]`

## [1.1.1] - 2022-04-03

### Added

- New native Python GUI, based on the Gooey package
- New standalone installer for Windows using PyInstaller and Inno Setup
- Added `deeplc-gui` entrypoint to start GUI from the command line

### Changed

- CLI: Restructured help message
- Made DeepLC class API docstring consistent with CLI help message
- Docs: Moved `dict_divider` and `split_cal` explanation to README Q&A section.
- CI: Only run tests on commits to `master` or from a PR
- Refactoring: Cleaned up `__main__.py`
- Logging: Changed some loggings from DEBUG to INFO level, some from WARNING to INFO or
  DEBUG level

### Removed

- Removed Java-based GUI in favor of new Python-based GUI

### Fixed

- If run through CLI/GUI, all Tensorflow warnings are now fully suppressed
- Added `--legacy_calibration` CLI option to allow for old piecewise linear calibration
  while new pyGAM calibration method is default. `--legacy_calibration` is mutually
  exclusive with `--pygam_calibration`.

## [1.0.1] - 2022-03-17

- Make version compatible with pip release

## [1.0.0] - 2022-02-23

- Make version compatible with pip release

## [1.0] - 2022-02-23

- Make pygam the default calibration method

## [0.2.2] - 2022-02-17

- Bug fix where split_cal was not correctly passed
- CMD support for pygam calibration

## [0.2.1] - 2022-02-03

- Scikit-learn and pygam dependency in setup

## [0.2.0] - 2022-01-21

- Bug fix duplicate peptide+mod for DeepCALLC
- New feature DeepCALLC

## [0.1.39] - 2022-01-12

- Version bump

## [0.1.38] - 2022-01-10

- Deep(CAL)LC functionality

## [0.1.37] - 2021-09-09

- Pygam as calibration function

## [0.1.36] - 2021-09-09

- Update to Streamlit webserver: Use `st.form` and new official download button

## [0.1.35] - 2021-08-09

- More elegant solution for library call as global

## [0.1.34] - 2021-07-29

- Fix var call dict object

## [0.1.33] - 2021-07-29

- Fix library delete call

## [0.1.32] - 2021-07-29

- Temporary fix suggested by markmipt for the library

## [0.1.31] - 2021-07-01

- Change logging to specified logger object instead of standard logger

## [0.1.30] - 2021-06-09

- GUI: Fix small font through starting jar with cmd to increase font size
- Added testing for Python 3.9
- Relax h5py requirement to allow v3
- Fixed GitHub Action workflow for Streamlit docker image build

## [0.1.29] - 2021-03-24

- Bug in writing library where a list was assumed so library only partially filled

## [0.1.28] - 2021-03-17

- Make it optional to reload library

## [0.1.27] - 2021-03-16

- Ignore library messages

## [0.1.26] - 2021-03-15

- Force older library h5py for compatability

## [0.1.25] - 2021-03-12

- Library gets appended for non-calibration peptides too

## [0.1.24] - 2021-03-12

- Change the default windows install to pip instead of conda
- Change reporting of identifiers used from library

## [0.1.23] - 2021-03-04

- Log the amount of identifiers in library used

## [0.1.22] - 2021-03-04

- Publish PyPI and GitHub release

## [0.1.21] - 2021-03-04

- Add library functionality that allows for storing and retrieving predictions (without running the model)

## [0.1.20] - 2021-02-19

- Describe hyperparameters and limit CPU threads
- Additional modfications, including those exclusive to pFind
- Change calibration error to warning (since it is a warning if it is out of range...)

## [0.1.18] - 2021-01-11

- Limit CPU usage by tensorflow by connecting to n_jobs

## [0.1.17] - 2020-08-07

- Support for Python 3.8

## [0.1.16] - 2020-05-18

- Bug fix in the calibration function
- Had to order the predicted instead of observed retention times of the calibration analytes
- Thanks to @courcelm for both finding and fixing the issue

## [0.1.15] - 2020-05-15

- Different calibration function, should not contain gaps anymore
- Changed to more accurate rounding
- Changed to splitting in groups of retention time instead of groups of peptides

## [0.1.14] - 2020-02-21

- Changed default model to a different data set

## [0.1.13] - 2020-02-21

- Duplicate peptides charge fix

## [0.1.12] - 2020-02-15

- Support for charges and spaces in peprec

## [0.1.11] - 2020-02-13

- Fixes in GUI

## [0.1.10] - 2020-02-10

- Include less models in package to meet PyPI 60MB size limitation

## [0.1.9] - 2020-02-09

- Bugfix: Pass custom activation function

## [0.1.8] - 2020-02-07

- Fixed support for averaging predictions of groups of models (ensemble) when no models were passed
- New models for ensemble

## [0.1.7] - 2020-02-07

- Support for averaging predictions of groups of models (ensemble)

## [0.1.6] - 2020-01-21

- Fix the latest release

## [0.1.5] - 2020-01-21

- Spaces in paths to files and installation allowed
- References to other CompOmics tools removed in GUI

## [0.1.5] - 2020-02-13

- Fixes in GUI

## [0.1.4] - 2020-01-17

- Fix the latest release

## [0.1.3] - 2020-01-17

- Fixed the .bat installer (now uses bioconda)

## [0.1.2] - 2019-12-19

- Example files in GUI folder
- Unnecesary bat and sh for running GUI removed

## [0.1.1] - 2019-12-18

- Switch to setuptools
- Reorder publish workflow; build wheels

## [0.1.1.dev8] - 2019-12-16

- Remove xgboost dependancy

## [0.1.1.dev7] - 2019-12-16

- Use dot instead of dash in versioning for bioconda

## [0.1.1-dev6] - 2019-12-15

- Fix publish action branch specification (2)

## [0.1.1-dev5] - 2019-12-15

- Fix publish action branch specification

## [0.1.1-dev4] - 2019-12-15

- Test other trigger for publish action

## [0.1.1-dev3] - 2019-12-15

- Update documentation, specify branch in publish action

## [0.1.1-dev2] - 2019-12-15

- Add long description to setup.py

## [0.1.1-dev1] - 2019-12-15

- Initial pre-release
