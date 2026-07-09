****************************
Migrating from DeepLC 3.x
****************************

DeepLC 4.0 is not backward compatible with v3. Update your code using the guide below.


Functional API replaces the ``DeepLC`` class
============================================

The ``DeepLC`` class is removed. Use the top-level functions from :mod:`deeplc` instead:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - v3
     - v4
   * - ``DeepLC().calibrate_preds(psm_list)``
     - :func:`deeplc.calibrate`
   * - ``DeepLC().make_preds(psm_list)``
     - :func:`deeplc.predict`
   * - *(no equivalent)*
     - :func:`deeplc.predict_and_calibrate` (predict + calibrate in one call)
   * - *(no equivalent)*
     - :func:`deeplc.finetune_and_predict` (fine-tune + predict in one call)

Example:

.. code-block:: python

   # v3
   from deeplc import DeepLC
   dlc = DeepLC(path_model="model.hdf5")
   dlc.calibrate_preds(psm_list_cal)
   preds = dlc.make_preds(psm_list)

   # v4
   from deeplc import predict_and_calibrate
   preds = predict_and_calibrate(psm_list, psm_list_reference=psm_list_cal)


Input format changes
====================

The legacy tab-separated format with ``seq`` and ``modifications`` columns is no longer
supported. Use any format supported by
`psm_utils <https://psm-utils.readthedocs.io/en/stable/>`_ instead, or a tab-separated
file with ``peptidoform`` and ``spectrum_id`` columns.

Peptide sequences now use
`ProForma 2.0 <https://pubs.acs.org/doi/10.1021/acs.jproteome.1c00771>`_ notation.
Modifications are encoded as bracketed labels (e.g. ``PEPTM[Oxidation]IDE`` or
``PEPTM[Formula:C1H2O]IDE``), not as a separate column.

See :ref:`Input file format` in the Usage guide for details.


Model checkpoints
=================

Legacy ``.hdf5`` checkpoints from v3 are not compatible with v4. The bundled model
has been retrained as a PyTorch multitask model (``multitask_model.pt``). Custom
``.hdf5`` checkpoints cannot be loaded; retrain using the v4 API. The new model
and should serve as an ideal starting point for fine-tuning to any custom setup.


Backend: TensorFlow → PyTorch
==============================

The deep learning backend changed from TensorFlow to PyTorch. TensorFlow is no longer
a dependency. GPU support uses PyTorch CUDA; install the appropriate PyTorch build from
`pytorch.org <https://pytorch.org/get-started/locally/>`_ if GPU acceleration is needed.
