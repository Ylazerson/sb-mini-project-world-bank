

# Springboard Mini Projects


----------



Project Organization
------------

```
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │    
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.    
    │    
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │        
        └── helper_functions.py  <- Scripts to assist in the project
```
    
    



--------
## Setup Instructions

---

### Setup conda:
```sh
conda create --name springboard-mini-projects python=3.6
conda activate springboard-mini-projects
```
---

### Install the requirements:
```sh
pip install -r requirements.txt
```
---

### Download the Kaggle competition files:
```sh
kaggle competitions download -c house-prices-advanced-regression-techniques
```
- This will place the files in the following dir: `~/.kaggle/competitions/house-prices-advanced-regression-techniques/` 
---


### Setup the IPython kernel:


```sh
python -m ipykernel install --user --name springboard-mini-projects --display-name "Python (springboard-mini-projects)"
```

---

## Walk through the mini-project notebooks:
```sh
cd notebooks
jupyter lab &
```   

