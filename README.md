
# AlgoHolics ML-Project

In this project we worked on the dataset for the kickstarter projects (https://www.kaggle.com/datasets/ulrikthygepedersen/kickstarter-projects/data).

In this .md we explain what you can find in this repo and how to work with it.
## How to work with the repo
**`GetData_and_EDA.ipynb`**

This data is the primary notebook of the repo. It includes:

- full EDA of the data **`kickstarter_projects.csv.zip`** from kaggle.com
- baseline model
- model comparison
    - feature engineering
    - different classifiers
    - precision metric for all models
- Limitations of the model

**`Models folder`**

Since the saved models are too big, they are not included in the repo. 

Code snippets are included, where the models get saved when the code ran (in the 'models' folder).

**Addition**

In the **branch 'plots'** there is also a performance test in the file **`kickstarter_projects.csv.zip`**, where it is tested, if there are structures in the wrong/right predicted data (none were found). 

## Presentation

The project presentation is saved as **`AlgoHolics_pres.pdf`**.

## Set up your Environment



### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


   
## Usage

In order to train the model and store test data in the data folder and the model in models run:

**`Note`**: Make sure your environment is activated.

```bash
python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.

