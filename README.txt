DEPENDENCIES
The project is written in Python 3.11. The required packages are listed in the "requirements.txt" file.

INSTALLATION
1. To avoid possible conflicts, it is recommended to use a **virtual environment** to install the required packages. 

    python3 -m venv datamining
    source datamining/bin/activate
    

2. To install the project, clone the repository and install the required packages with the following command:
    
    pip install -r requirements.txt


HOW TO RUN THE PROGRAM

The program can be run with the following command:

    - Default and recommended way:
        python3 egmMain.py -s <standard-routes-file> -a <actual-routes-file>

    - Complete and tunable way:
        python3 egmMain.py -s <standard-routes-file> -a <actual-routes-file> -p -n <num-plots> -k <k-shingles> -m <metric> -f <fusion-method> --alpha <weight>

        where:
            -s: standard routes file
            -a: actual routes file
            -p: plot the results
            -n: number of plots to show
            -k: number of shingles
            -m: metric to use for merchandise (jaccard, cosine, L2)
            -f: fusion method to use between route and merchandise similarity
            --alpha: weight (between 0 and 1) of the route similarity  wrt the merchandise similarity

        - Help:
            python3 egmMain.py -h
        
    - Fast but maximum dataset size 10_000:
        python3 egmMainFast.py (... same parameters as above)

NB: the standard route file must start with "standard" and the actual route file must start with "actual".