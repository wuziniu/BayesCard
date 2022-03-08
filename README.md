# BayesCard

## Environment setup
  The following command using conda should setup the environment in linux CentOS.
  ```
  conda env create -f environment.yml
  ```
  If not, you need to manually download the following packages
  Required dependence: numpy, scipy, pandas, Pgmpy, pomegranate, networkx, tqdm, joblib, pytorch, psycopg2, scikit-learn, 
  Additional dependence: numba, bz2, Pyro (These packages are not required to reproduce the result in the paper.)
  
## Dataset download:
The optimal trained models for each dataset are already stored. If you are only interested in verifying the paper's result, you can skip the dataset download and model training, and directly execute the evaluate the learnt model.
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
         --csv_path .../DMV/DMV.csv
         --model_path Benchmark/DMV
         --learning_algo chow-liu
         --max_parents 1
         --sample_size 200000
  ```
  model_path specifies the location to save the model
  
  csv_path points the dataset you just downloaded
  
  algo: one can choose between chow-liu, greedy, exact, junction. Expect for chow-liu, other methods contain a large amount of randomness, so not garuantee to 
  reproduce the exactly same result as paper.
  
  Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset dmv
         --evaluate_cardinalities
         --model_path Benchmark/DMV/chow-liu_1.pkl
         --query_file_location Benchmark/DMV/query.sql
         --infer_algo exact-jit
  ```
  infer_algo: one can choose between exact, exact-jit, exact-jit-torch, progressive_sampling, BP and sampling. I'm current working on BP's optimization, so there might be some unexpected bugs. 
  
## Reproducing Census result:
  Similar to DMV, first train the model
  ```
  python run_experiment.py --dataset census
         --generate_models
         --csv_path ../Census/Census.csv
         --model_path Benchmark/Census
         --learning_algo chow-liu
         --max_parents 1
         --sample_size 200000
  ```
  Then, evaluate the learnt model
  ```
  python run_experiment.py --dataset census
         --evaluate_cardinalities
         --model_path Benchmark/Census/chow-liu_1.pkl
         --query_file_location Benchmark/Census/query.sql
         --infer_algo exact-jit
  ```
  
## Reproducing stability and scalability experiment on synthetic datasets:
   Please refer to jupyter notebook: Testing/stability_experiment.ipybn, which contains the step-by-step guide to reproduce the result.
   Other notebooks in that directory are for debug purposes only.

## Reproducing IMDB results:
   First prepare the data, i.e. adding some fanout columns
   ```
   python run_experiment.py --dataset imdb 
          --generate_hdf 
          --csv_path ../imdb-benchmark
          --hdf_path ../imdb-benchmark/gen_hdf
   ```
   Then train a Bayescard ensemble of BNs
   ```
   python run_experiment.py --dataset imdb 
          --generate_models
          --hdf_path ../imdb-benchmark/gen_hdf
          --model_path Benchmark/IMDB
          --learning_algo chow-liu
          --max_parents 1
          --sample_size 200000
   ```
   Evaluate the learnt Bayescard
   ```
   python run_experiment.py --dataset imdb 
          --evaluate_cardinalities
          --model_path Benchmark/IMDB/
          --query_file_location Benchmark/IMDB/job-light.sql
          --learning_algo chow-liu
          --max_parents 1
          --infer_algo exact-jit
   ```
  

