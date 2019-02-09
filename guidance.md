# Details for running AGen

## Setup Environment 
- install dependancy and data preprocessing
see full_preprocessing_demon.md
please modify dataset file name in validate_utils.py & adaption.py

- Install AGen
clone files in this reporsitory directly to YOURPATH/ngsim_env/scripts/imitation

- Pretrained model and hyperparameters
copy .npz files in ./pretrained/ to YOURPATH/ngsim_env/data/experiments/multiagent_curr/imitate/log/

## Run Code

```bash
# Train and run a single agent adaptive algorithm
python adaption.py --n_proc 1 --exp_dir ../../data/experiments/multiagent_curr/ --params_filename itr_200.npz --use_multiagent True --n_envs 22 --adapt_steps 1(or2) 
# Train and run a single/multi adaptive algorithm
python adaption.py --n_proc 1 --exp_dir ../../data/experiments/multiagent_curr/ --params_filename itr_200.npz --use_multiagent False --n_envs 1 --adapt_steps 1(or2)

```

## Supporting files
- theta.npy 
extracted top layer for pretrained RNN, use as initialization

- check_convergence.py
check whether pretrained GAN is valid


## Data Generation
output .npz file will be in YOURPATH/ngsim_env/data/experiments/multiagent_curr/imitate
