# Data Mining Project 2023/2024

Project for the Data Mining course of the Master's Degree in Artificial Intelligence Systems at the University of Trento, A.Y. 2023/2024.

## Description
The project consists of designing and implementing a data mining solution for a real-world application. The goal is to find the best recommendation system for a logistic company that wants to optimize the cost of its transportation service by predicting the best route to assign to each truck driver in its fleet.
For more details, please refer to the [project description](DM_Project.pdf).

## Dependencies
The project is written in Python 3.11. The required packages are listed in the [requirements](requirements.txt) file.

## Installation
1. To avoid possible conflicts, it is recommended to use a **virtual environment** to install the required packages. 
    ```
    python3 -m venv datamining
    source datamining/bin/activate
    ``` 

2. To install the project, clone the repository and install the required packages with the following command:
    ```
    pip install -r requirements.txt
    ```

## How to run the program

The program can be run with the following command:

- Default and recommended way:
    ```
    python3 egmMain.py -s <standard-routes-file> -a <actual-routes-file>
    ```

- Complete and tunable way:
    ```
    python3 egmMain.py -s <standard-routes-file> -a <actual-routes-file> -p -n <num-plots> -k <k-shingles> -m <metric> -f <fusion-method> --alpha <weight>
    ```
    where:
    - `-s`: standard routes file, must start with "standard"
    - `-a`: actual routes file, must start with "actual"
    - `-p`: plot the results
    - `-n`: number of plots to show (4 by default), -p is required
    - `-k`: number of shingles, greater than 2
    - `-m`: metric to use for merchandise (jaccard, cosine, L2)
    - `-f`: fusion method to use between route and merchandise similarity
    - `--alpha`: weight (between 0 and 1) of the route similarity  wrt the merchandise similarity

- Help:
    ```
    python3 egmMain.py -h
    ```

**Note:** The standard route file must start with "standard" and the actual route file must start with "actual".

## Authors
Mattia Nardon, Giovanni Scialla, Eric Suardi.