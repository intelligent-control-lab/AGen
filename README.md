This is the implementation of adaptive algorithm on PSGAIL.
The algorithm is based on NGSIM Env, which is a rllab environment for learning human driver models with imitation learning. 
# Installation Process
Step-by-step install instructions are at [`docs/install_env_gail_full.md`](docs/install_env_gail_full.md)
# Train and run a single/multi-agent adaptive algorithm
```bash
python adaption.py --n_proc 1 --exp_dir ../../data/experiments/multiagent_curr/ --params_filename itr_200.npz --use_multiagent True --n_envs 22 --adapt_steps 1(or2) 
python adaption.py --n_proc 1 --exp_dir ../../data/experiments/multiagent_curr/ --params_filename itr_200.npz --use_multiagent False --n_envs 1 --adapt_steps 1(or2)

```
