# Medication classification from EEG
> Software implementing methods from: David O. Nahmias, and Kimberly L. Kontson. Deep Learning and Feature Based Medication Classifications from EEG in a Large Clinical Data Set. In review, 2020.

This work evaluates feature-based and deep learning based methods to classify medication statuses of patients using solely EEG.

## Usage
For feature-based learning:
```sh
python sortMedications.py
```
with the desired model selected and files in _data_ subfolder.

For deep learning based models:
```sh
./deepShell
```
with the desired model selected, scripts in shared folder, and files in _data_ subfolder.


## Development setup

Feature-based learning developed and tested on Python 2.7, packages used found in software header.
Deep learning based models developed and tested on Python 3.5, packages used found in software header.

Make file coming soon.


## Release History

* 0.0.1
    * Initial commit of stable project

## Meta

David Nahmias – [Website](dnahmias.com) – david.nahmias@fda.hhs.gov

Distributed under the public domain license. See ``LICENSE`` for more information.

[https://github.com/dbp-osel](https://github.com/dbp-osel/)


## Citation
David O. Nahmias, and Kimberly L. Kontson. Deep Learning and Feature Based Medication Classifications from EEG in a Large Clinical Data Set. In review, 2020.
