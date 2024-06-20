# 3D-Medical-DDPM

**Author**: Andrea Basile

**Email**: andreabasile97@gmail.com

**Version**: 0.0

## Technical requirements before installing

This project needs C++ v.14.0.0 or greater and 
Python 3.8
cuda 12.1

DDPM for 3D medical image synthesis

## Repository structure
```
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── docs               <- Directory for documentation.
├── models             <- Trained and serialized models, model predictions, or model summaries.
├── references         <- Directory containing project references.
├── reports            <- Generated HTML, PDF, LaTeX, etc. reports.
│   └── figures        <- Graphs and figures generated to use in reports.
├── src                <- Project source code.
│   ├── data           <- Scripts to download or generate data.
│   ├── features       <- Scripts to turn raw data into features.
│   ├── models         <- Scripts to train and use models.        
│   └── visualization  <- Scripts to create visualizations.
├── test               <- Project test code.
│   ├── data           
│   ├── features       
│   ├── models             
│   └── visualization  
├── Makefile           <- Makefile with `make install_requirements` command.
├── README.md          <- Project markdown file created.
├── requirements.txt   <- Txt file containing all requirements to install in venv.
├── setup.sh           <- Allows you to configure git and DVC.
```