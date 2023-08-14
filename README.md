# CSIFNet：Deep multimodal network based on cellular spatial information fusion for survival prediction.
Using multimodal data to construct models for cancer survival prediction is essential for the prognosis of patients.   Most of existing methods do not consider the cellular spatial organization of tumors at the single-cell level. In this work, we present CSIFNet, a deep multimodal network based on cellular spatial information fusion. The multimodal data used includes genomics data, clinical variables data and cellular spatial data.    Specifically, we construct a spatial information fusion module (SIFM) to learn the local interactions between tumor cells and the tumor microenvironment to obtain a more comprehensive high-level representation of tumor cellular spatial information.   Experimental results show that our proposed method has better performance than the state-of-the-art survival prediction methods.

# Dependencies
This source code requires the key dependencies as follows:
'''python
python 3.7.13
torch==1.7.1
numpy==1.19.5
'''
\\
You can run this following command to install additional dependencies.\\
`pip install -r requirements.txt`
# Data
The METABRIC Imaging mass cytometry data of tumor cell dataset may be obtained from https://idr.openmicroscopy.org/ (accession code idr0076). Imaging mass cytometry data of tumor mircoenvirment may be obtained from https://doi.org/10.5281/zenodo.5850952. METABRIC genomic and clinical data may be obtained from cBioPortal (https://www.cbioportal.org/). METABRIC genomic data are available from the European Genome-phenome Archive under accession numbers EGAS00000000083 and EGAS00001001753.

# Experiments
Basic configurations for all hyperparameters in `config2.py`.\\
To train the CSIFNet, you can run the folling command.\\
```python
python train.py --fold_id = ${fold_id}
```

