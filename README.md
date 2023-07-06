# Decision Tree based - Hindsight Experience Replay (DT-HER)

This is the implementation for our paper "Generating Curriculum with Decision Tree under the Sparse Reward".

Also contains a TensorFlow implementation for the paper [Exploration via Hindsight Goal Generation](http://arxiv.org/abs/1906.04279) accepted by NeurIPS 2019.



## Requirements
1. Python 3.6.9
2. MuJoCo == 1.50.1.68
3. TensorFlow >= 1.8.0
4. BeautifulTable == 0.7.0
5. gym < 0.22

## Running Commands

Run the following commands to reproduce our results shown in section 6.2.

```bash
python train.py --tag='DT-HER_fetch_push' --learn=dt-her --env=FetchPush-v1 --goal=interval
python train.py --tag='DT-HER_fetch_slide' --learn=dt-her --env=FetchSlide-v1 --goal=interval
python train.py --tag='DT-HER_fetch_reach' --learn=dt-her --env=FetchReach-v1 --goal=interval
python train.py --tag='DT-HER_fetch_pick' --learn=dt-her --env=FetchPickAndPlace-v1 --goal=interval

python train.py --tag='DT-HER_fetch_push_with_obstacle' --learn=dt-her --env=FetchPush-v1 --goal=obstacle
python train.py --tag='DT-HER_fetch_slide_with_obstacle' --learn=dt-her --env=FetchSlide-v1 --goal=obstacle
python train.py --tag='DT-HER_fetch_pick_with_obstacle' --learn=dt-her --env=FetchPickAndPlace-v1 --goal=obstacle
```
