# Evaluation of Transaction Data Analysis

This document provides instructions for evaluating the performance of an algorithm based on transaction data. The evaluation involves running an evaluation script on macOS.

## Pre-requisites
- Python 3.x installed on your macOS system.
- Basic knowledge of working with the terminal.

## Usage Instructions


### 1. Set Up Virtual Environment
Navigate to the project directory in your terminal and create a virtual environment:
``` 
cd transaction-analysis
python3 -m venv venv
``` 

### 2. Install Required Packages
With the virtual environment activated, install the required packages:
``` 
pip install -r requirements.txt
```

### 3. Run Evaluation Script
To evaluate the performance of the algorithm, run the evaluation script:
```
python evaluation_script.py
```
### 4. Provide File Path
When prompted, provide the path to the CSV file containing the transaction data. Ensure the CSV file is located in the project directory or specify the full path.

### Additional Information
All the training processes are included in the training_model.ipynb notebook.
The evaluation script is named evaluation_script.py.
A requirements.txt file is provided for installing the necessary Python packages.
If required, you can create a virtual environment and install the packages to run the evaluation script.

