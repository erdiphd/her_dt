# Evolutionary learning of interpretable decision trees 
This repo hosts the code of the paper [L. L. Custode and G. Iacca, Evolutionary learning of interpretable decision trees. 2020](https://arxiv.org/abs/2012.07723).

``@misc{custode2020evolutionary,
      title={Evolutionary learning of interpretable decision trees},
      author={Leonardo Lucio Custode and Giovanni Iacca},
      year={2020},
      eprint={2012.07723},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}``

## Installation
This implementation uses the [DEAP](https://github.com/DEAP/deap) library.
To install all the requirements use `pip install -r requirements.txt`

## Summary of the files
### simple_test_orthogonal.py
This file contains the implementation used to evolve the agents for the CartPole-v1 and MountainCar-v0 environments with the orthogonal grammar.
Of course you can use it in any OpenAI Gym environment that has a discrete action set.
You can repeat the experiments by executing:

```
python3 simple_test_orthogonal.py --environment_name CartPole-v1 --jobs <n_jobs> --seed <seed> --n_actions 2 --learning_rate 0.001 --df 0.05 --input_space 4 --episodes 10 --lambda_ 200 --generations 100 --cxp 0 --mp 1 --low -1 --up 1 --types '#-48,48,5,10;-50,50,5,10;-418,418,5,1000;-836,836,5,1000' --mutation "function-tools.mutUniformInt#low-0#up-40000#indpb-0.1"
```
```
python3 simple_test_orthogonal.py --environment_name MountainCar-v0 --jobs <n_jobs> --seed <seed> --n_actions 3 --learning_rate 0.001 --df 0.05 --input_space 2 --episodes 10 --lambda_ 200 --generations 1000 --cxp 0 --mp 1 --low -1 --up 1 --types '#-120,60,5,100;-70,70,5,1000' --mutation "function-tools.mutUniformInt#low-0#up-40000#indpb-0.05"
```

### advanced_test_oblique.py
This file contains the implementation of our method with an oblique grammar.
Also in this case, you can use it with any OpenAI Gym environment that has a discrete action set.

You can repeat the experiments (e.g. on LunarLander-v2) by executing:
```
python3 advanced_test_oblique.py --crossover 'function-tools.cxOnePoint' --cxp 0.1 --decay 0.99 --df 0.9 --environment_name 'LunarLander-v2' --episode_len 1000 --episodes 1000 --eps 1.0 --generations 100 --genotype_len 100 --input_space 8 --jobs 20 --lambda_ 100 --learning_rate 'auto' --mp 1.0 --mutation 'function-tools.mutUniformInt#low-0#up-4000#indpb-0.05 --n_actions 4 --patience 30 --seed 4 --selection 'function-tools.selTournament#tournsize-2 --timeout 600 --types '#-000,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000;-00,1001,1000,1000'
```

# Notes
The logs for the MountainCar environment contain approximation issues: in fact, since some values in the leaves have been approximated and the argmax function returns the first action that corresponds to the max, some policies may have issues when tested.
Fixing these approximation issues makes the agents work as intended, as shown by the scores in the paper.
