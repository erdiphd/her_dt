import os
from random import randrange

import gym
import json
import string
import datetime
import argparse
import subprocess
import numpy as np
from time import time, sleep
from numpy import random
from ge_q_dts.dt import EpsGreedyLeaf, PythonDT, RandomlyInitializedEpsGreedyLeaf
from ge_q_dts.grammatical_evolution import GrammaticalEvolutionTranslator, grammatical_evolution, differential_evolution
import gym_examples

def string_to_dict(x):
    """
    This function splits a string into a dict.
    The string must be in the format: key0-value0#key1-value1#...#keyn-valuen
    """
    result = {}
    items = x.split("#")

    for i in items:
        key, value = i.split("-")
        try:
            result[key] = int(value)
        except:
            try:
                result[key] = float(value)
            except:
                result[key] = value

    return result


parser = argparse.ArgumentParser()
parser.add_argument("--jobs", default=1, type=int, help="The number of jobs to use for the evolution")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--environment_name", default="gym_examples:gym_examples/GridWorld-v0", help="The name of the environment in the OpenAI Gym framework")
parser.add_argument("--n_actions", default=4, type=int, help="The number of action that the agent can perform in the environment")
parser.add_argument("--learning_rate", default="auto", help="The learning rate to be used for Q-learning. Default is: 'auto' (1/k)")
parser.add_argument("--df", default=0.9, type=float, help="The discount factor used for Q-learning")
parser.add_argument("--eps", default=0.05, type=float, help="Epsilon parameter for the epsilon greedy Q-learning")
parser.add_argument("--input_space", default=2, type=int, help="Number of inputs given to the agent")
parser.add_argument("--num_episodes", default=50, type=int, help="Number of episodes that the agent faces in the fitness evaluation phase")
parser.add_argument("--episode_len", default=300, type=int, help="The max length of an episode in timesteps")
parser.add_argument("--lambda_", default=50, type=int, help="Population size")
parser.add_argument("--generations", default=200, type=int, help="Number of generations")
parser.add_argument("--cxp", default=0.5, type=float, help="Crossover probability")
parser.add_argument("--mp", default=0.5, type=float, help="Mutation probability")
parser.add_argument("--mutation", default="function-tools.mutUniformInt#low-0#up-40000#indpb-0.1", type=string_to_dict, help="Mutation operator. String in the format function-value#function_param_-value_1... The operators from the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default: Uniform Int Mutation")
parser.add_argument("--crossover", default="function-tools.cxOnePoint", type=string_to_dict, help="Crossover operator, see Mutation operator. Default: One point")
parser.add_argument("--selection", default="function-tools.selTournament#tournsize-2", type=string_to_dict, help="Selection operator, see Mutation operator. Default: tournament of size 2")

parser.add_argument("--genotype_len", default=100, type=int, help="Length of the fixed-length genotype")
parser.add_argument("--low", default=-10, type=float, help="Lower bound for the random initialization of the leaves")
parser.add_argument("--up", default=10, type=float, help="Upper bound for the random initialization of the leaves")
parser.add_argument("--types", default="#0,20,1,1;0,20,1,1", type=str, help="This string must contain the range of constants for each variable in the format '#min_0,max_0,step_0,divisor_0;...;min_n,max_n,step_n,divisor_n'. All the numbers must be integers.")

# add arguments from common.py. they are not affecting anything, just that we can run train py with arguments from command line
#------------------------------------------------------------------------------------------
parser.add_argument('--tag', help='terminal tag in logger', type=str, default='')
parser.add_argument('--alg', help='backend algorithm', type=str, default='ddpg', choices=['ddpg', 'ddpg2'])
parser.add_argument('--learn', help='type of training method', type=str, default='normal')

parser.add_argument('--env', help='gym env id', type=str, default='FetchPickAndPlace-v1')
args, _ = parser.parse_known_args()
if args.env == 'HandReach-v0':
    parser.add_argument('--goal', help='method of goal generation', type=str, default='reach',
                        choices=['vanilla', 'reach'])
else:
    parser.add_argument('--goal', help='method of goal generation', type=str, default='obstacle',
                        choices=['vanilla', 'fixobj', 'interval', 'obstacle'])
    if args.env[:5] == 'Fetch':
        parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32,
                            default=1.0)
    elif args.env[:4] == 'Hand':
        parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32,
                            default=0.25)

parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.98)
parser.add_argument('--clip_return', help='whether to clip return value', type=bool, default=True)
parser.add_argument('--eps_act', help='percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
parser.add_argument('--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32,
                    default=0.2)

parser.add_argument('--pi_lr', help='learning rate of policy network', type=np.float32, default=1e-3)
parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=1e-3)
parser.add_argument('--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
parser.add_argument('--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32,
                    default=0.95)

parser.add_argument('--epochs', help='number of epochs', type=np.int32, default=20)
parser.add_argument('--cycles', help='number of cycles per epoch', type=np.int32, default=20)
parser.add_argument('--episodes', help='number of episodes per cycle', type=np.int32, default=50)
parser.add_argument('--timesteps', help='number of timesteps per episode', type=np.int32,
                    default=(50 if args.env[:5] == 'Fetch' else 100))
parser.add_argument('--train_batches', help='number of batches to train per episode', type=np.int32, default=20)

parser.add_argument('--buffer_size', help='number of episodes in replay buffer', type=np.int32, default=10000)
parser.add_argument('--buffer_type', help='type of replay buffer / whether to use Energy-Based Prioritization',
                    type=str, default='energy', choices=['normal', 'energy'])
parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=256)
parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)
parser.add_argument('--her', help='type of hindsight experience replay', type=str, default='future',
                    choices=['none', 'final', 'future'])
parser.add_argument('--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full',
                    choices=['full', 'final'])

parser.add_argument('--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
parser.add_argument('--hgg_L', help='Lipschitz constant', type=np.float32, default=5.0)
parser.add_argument('--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)
parser.add_argument('--forced_hgg_dt_step_size', help='step size between intermediate goals', type=np.float32,
                    default=None)
parser.add_argument('--c', help='c parameter to measure success basing on Q function', type=np.float32, default=-1)
parser.add_argument('--save_acc', help='save successful rate', type=bool, default=True)

# Setup of the logging

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = "ge_q_dts/logs/gym/{}_{}".format(date, "".join(np.random.choice(list(string.ascii_lowercase), size=11)))
logfile = os.path.join(logdir, "log.txt")
os.makedirs(logdir)

args = parser.parse_args()

best = None
input_space_size = args.input_space
lr = "auto" if args.learning_rate == "auto" else float(args.learning_rate)


# Creation of an ad-hoc Leaf class

class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    def __init__(self):
        super(CLeaf, self).__init__(args.n_actions, lr, args.df, args.eps, low=args.low, up=args.up)


# Setup of the grammar

grammar = {
    "bt": ["<if>"],
    "if": ["if <condition>:{<action>}else:{<action>}"],
    "condition": ["_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(input_space_size)],
    "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
    "comp_op": [" < ", " > "],
}

types = args.types if args.types is not None else ";".join(["0,10,1,10" for _ in range(input_space_size)])
types = types.replace("#", "")
assert len(types.split(";")) == input_space_size, "Expected {} types, got {}.".format(input_space_size, len(types.split(";")))

for index, type_ in enumerate(types.split(";")):
    rng = type_.split(",")
    start, stop, step, divisor = map(int, rng)
    consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)]))
    grammar["const_type_{}".format(index)] = consts_

print(grammar)


# Seeding of the random number generators

random.seed(args.seed)
np.random.seed(args.seed)


# Log all the parameters

with open(logfile, "a") as f:
    vars_ = locals().copy()
    for k, v in vars_.items():
        f.write("{}: {}\n".format(k, v))


# Definition of the fitness evaluation function

def evaluate_fitness(fitness_function, leaf, genotype, grid_size, agent_start, agent_goal, dimensions, reward_type, episodes=args.num_episodes):
    phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(genotype)
    bt = PythonDT(phenotype, leaf)
    return fitness_function(bt, grid_size, agent_start, agent_goal, dimensions, reward_type, episodes)


def fitness(x, grid_size, agent_start, agent_goal, dimensions, reward_type, obstacle_is_on, episodes=args.num_episodes):
    random.seed(args.seed)
    np.random.seed(args.seed)
    global_cumulative_rewards = []
    environment_name = "gym_examples/GridWorld-v0"
    e = gym.make(environment_name, size=grid_size, agent_location=agent_start, target_location=agent_goal,
                 dimensions=dimensions, reward_type=reward_type, obstacle_is_on=obstacle_is_on)

    try:
        for iteration in range(episodes):
            # e.seed(iteration)
            obs = e.reset()[0]
            x.new_episode()
            cum_rew = 0
            previous = None
            for t in range(args.episode_len):

                action = x(obs['agent'])

                obs, rew, done, truncated, info = e.step(action)

                # e.render()
                x.set_reward(rew)

                cum_rew += rew

                if done:
                    break

            x.set_reward(rew)

            x(obs['agent'])
            # x(obs['observation'])
            global_cumulative_rewards.append(cum_rew)
    except Exception as ex:
        # print(ex)
        if len(global_cumulative_rewards) == 0:
            global_cumulative_rewards = -1000
    e.close()

    fitness = np.mean(global_cumulative_rewards),
    return fitness, x.leaves


# if __name__ == '__main__':
def main(grid_size, agent_start, agent_goal, dimensions, reward_type, obstacle_is_on):

    import collections
    from joblib import parallel_backend

    def fit_fcn(x):
        return evaluate_fitness(fitness, CLeaf, x, grid_size, agent_start, agent_goal, dimensions, reward_type, obstacle_is_on)

    with parallel_backend("multiprocessing"):
        pop, log, hof, best_leaves = grammatical_evolution(fit_fcn, inputs=input_space_size, leaf=CLeaf, individuals=args.lambda_, generations=args.generations, jobs=args.jobs, cx_prob=args.cxp, m_prob=args.mp, logfile=logfile, seed=args.seed, mutation=args.mutation, crossover=args.crossover, initial_len=args.genotype_len, selection=args.selection)
        # returning evolved structure of decision tree

    # Log best individual

    with open(logfile, "a") as log_:
        phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(hof[0])
        phenotype = phenotype.replace('leaf="_leaf"', '') # phenotype : decision tree, genotype: selecting rules from grammar

        for k in range(50000):  # Iterate over all possible leaves
            key = "leaf_{}".format(k)
            if key in best_leaves:
                v = best_leaves[key].q
                phenotype = phenotype.replace("out=_leaf", "out={}".format(np.argmax(v)), 1) # choose best action from best_leaves
            else:
                break

        log_.write(str(log) + "\n")
        log_.write(str(hof[0]) + "\n")
        log_.write(phenotype + "\n")
        log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))
    with open(os.path.join(logdir, "fitness.tsv"), "w") as f:
        f.write(str(log))
    # return DT as string
    return phenotype

