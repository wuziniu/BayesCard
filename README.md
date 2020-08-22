### BayesCard

## Environment setup
  The following command should setup the environment in linux CentOS.
  ```
  Conda: conda env create -f environment.yml
  ```
  If not, you need to manually download the following packages
  Required dependence: numpy, scipy, pandas, Pgmpy, pomegranate, networkx, tqdm, joblib, 
  Additional dependence: numba, bz2, Pyro
  
## Dataset download:
1. DMV dataset:
   The DMV dataset is publically available at catalog.data.gov. The data is continuously updated but there shouldn't be any shift in distribution. If you would
   like to reproduce the exact result as in paper, please contact the author for the snapshot that he downloaded.

2. Census dataset:
   The census dataset is publically available on UCI website: https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29.

3. IMDB dataset:
   The imdb dataset can be downloaded here: http://homepages.cwi.nl/~boncz/job/imdb.tgz
   
## Reproducing DMV result:
  In order to reproduce the single table result for DMV, 
  First run the following command to train the model
  ```
  python run_experiment.py --dataset dmv
         --generate_models
         --model_path ../Benchmark/DMV
         --algo chow-liu
         --max_parents 1
         --sample_size 200000
  ```
  model_path specifies the location to save the model
  algo: one can choose between chow-liu, greedy, exact, junction. Expect for chow-liu, other methods contain a large amount of randomness, so not garuantee to 
  reproduce the exactly same result as paper.
  
  Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset dmv
         --evaluate_cardinalities
         --model_location ../Benchmark/DMV/chow-liu_1.pkl
         --query_file_location ../Benchmark/DMV/query.sql
  ```
