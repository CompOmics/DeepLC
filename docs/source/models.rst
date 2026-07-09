*****************
Prediction models
*****************

Default model
=============

DeepLC 4.0 ships a pretrained multitask model (``multitask_model.pt``) as the
default. This model was trained jointly across multiple LC setups and outputs
one retention time prediction per setup. The best-fitting output head is
selected automatically during calibration based on Pearson correlation to the
observed retention times in the reference set.

Training a model from scratch
==============================

:func:`deeplc.train` trains a new model on a PSM list with observed retention
times:

.. code-block:: python

   from psm_utils.io import read_file
   from deeplc import train, save_model

   psm_list = read_file("training_psms.tsv")
   model = train(psm_list)
   save_model(model, "my_model.pt")

Using a custom model
====================

While the default model should work well for nearly all LC setups, a custom
model checkpoint can be passed to any core function via the ``model`` argument:

.. code-block:: python

   from deeplc import predict_and_calibrate

   calibrated_rt = predict_and_calibrate(psm_list, model="path/to/model.pt")

Checkpoints must be plain PyTorch state dicts saved with
``torch.save(model.state_dict(), path)``. See :func:`deeplc.save_model`.
