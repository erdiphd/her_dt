#!/bin/bash

source /home/user/conda/bin/activate her_dt
cd /home/user/her_dt/

pip install -e /home/user/her_dt/gym-examples
cd /home/user/her_dt/
sudo chown -R user:user /home/user/her_dt

python train.py
# python train.py --tag=${tag} --learn=${learn} --env=${env} --goal=${goal} --epochs=${epochs} --cycles=${cycles} --episodes=${episodes} --obstacle=${obstacle} --forced_hgg_dt_step_size=${forced_hgg_dt_step_size}
