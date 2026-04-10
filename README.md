# TCR-GACGen
An antigen-specific TCR generation model conditioned with single-cell transcriptome
## Model Architecture
![workflow](https://github.com/gaol00034/TCR-GACGen/blob/main/Figures/Model.png)
***
## Data description
All the TCR-pMHC data are download from [VDJdb](https://vdjdb.cdr3.net/overview), [IEDB](https://www.iedb.org/) and [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR); 

All the TCR-GEX data are download from [huARdb](https://huarc.net/v2/);

You can download the models from [ckpt](https://drive.google.com/file/d/1o4RPr2pJK8F_bT5EuUNp-MF2tScadO5T/view?usp=drive_link).
***
## Run

For generating the TCRs(CDR3bs) with given pMHCs and GEX, please run the script "./Scripts/inference.py";

The input data format could be refered from in "./data/inputsample.csv";
***
## Contact
If you have any questions, please contact us at [ltgao34@njust.edu.cn](ltgao34@njust.edu.cn) or [njyudj@njust.edu.cn](njyudj@njust.edu.cn)
