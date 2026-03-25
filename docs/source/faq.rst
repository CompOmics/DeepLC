**************************
Frequently asked questions
**************************

Is it required to indicate fixed modifications in the input file?
=================================================================

Yes, even modifications like carbamidomethyl should be in the input file.


So DeepLC is able to predict the retention time for any modification?
=====================================================================

Yes, DeepLC can predict the retention time of any modification. However, if the
modification is **very** different from the peptides the model has seen during
training the accuracy might not be satisfactory for you. For example, if the model
has never seen a phosphor atom before, the accuracy of the prediction is going to
be low.


I have a special usecase that is not supported. Can you help?
=============================================================

Of course, please feel free to contact us:

Robbin.Bouwmeester@UGent.be and Ralf.Gabriels@UGent.be


DeepLC runs out of memory. What can I do?
==========================================

You can try to reduce the batch size. DeepLC should be able to run if the batch size is low
enough, even on machines with only 4 GB of RAM.


I have a GPU, but DeepLC is not using it. Why?
==============================================================

Ensure that you have the correct version of PyTorch installed. See the PyTorch
[Getting Started](https://pytorch.org/get-started/locally/) page for more information.


What modification name should I use?
=====================================

Amino acid modification labels must be resolvable to a known chemical formula. This means that
accepted labels are:

- A name or accession from an controlled vocabulary, such as Unimod or PSI-MOD. (e.g., Oxidation, U:Oxidation, U:35, MOD:00046…)
- An elemental formula (e.g, Formula:C12H20O2)

When the modification label is not resolvable to a known chemical formula, for instance when using
a mass shift notation, the modification will be ignored and the prediction will be based on the
unmodified peptide.


I have a modification that is not in Unimod or PSI-MOD. How can I add the modification?
========================================================================================

Custom modifications can be added by using the elemental formula of the modification.

For example: ``SEQUEN[Formula:C12H20O2]CE``
