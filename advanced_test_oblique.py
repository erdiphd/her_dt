import os
import gym
import json
import string
import datetime
import stopit
import argparse
import subprocess
import numpy as np
from numpy import random
from time import time, sleep
from matplotlib import pyplot as plt
from multiprocessing import TimeoutError
from dt import EpsGreedyLeaf, PythonDT, RandomlyInitializedEpsGreedyLeaf
from grammatical_evolution import GrammaticalEvolutionTranslator, grammatical_evolution, differential_evolution


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
            result[key] = float(value)
        except:
            try:
                result[key] = int(value)
            except:
                result[key] = value

    return result


parser = argparse.ArgumentParser()
parser.add_argument("--jobs", default=1, type=int, help="The number of jobs to use for the evolution")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--environment_name", default="LunarLander-v2", help="The name of the environment in the OpenAI Gym framework")
parser.add_argument("--n_actions", default=4, type=int, help="The number of action that the agent can perform in the environment")
parser.add_argument("--learning_rate", default="auto", help="The learning rate to be used for Q-learning. Default is: 'auto' (1/k)")
parser.add_argument("--df", default=0.9, type=float, help="The discount factor used for Q-learning")
parser.add_argument("--eps", default=0.05, type=float, help="Epsilon parameter for the epsilon greedy Q-learning")
parser.add_argument("--input_space", default=8, type=int, help="Number of inputs given to the agent")
parser.add_argument("--episodes", default=50, type=int, help="Number of episodes that the agent faces in the fitness evaluation phase")
parser.add_argument("--episode_len", default=1000, type=int, help="The max length of an episode in timesteps")
parser.add_argument("--lambda_", default=30, type=int, help="Population size")
parser.add_argument("--generations", default=1000, type=int, help="Number of generations")
parser.add_argument("--cxp", default=0.5, type=float, help="Crossover probability")
parser.add_argument("--mp", default=0.5, type=float, help="Mutation probability")
parser.add_argument("--mutation", default="function-tools.mutUniformInt#low-0#up-40000#indpb-0.1", type=string_to_dict, help="Mutation operator. String in the format function-value#function_param_-value_1... The operators from the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default: Uniform Int Mutation")
parser.add_argument("--crossover", default="function-tools.cxOnePoint", type=string_to_dict, help="Crossover operator, see Mutation operator. Default: One point")
parser.add_argument("--selection", default="function-tools.selTournament#tournsize-2", type=string_to_dict, help="Selection operator, see Mutation operator. Default: tournament of size 2")
parser.add_argument("--genotype_len", default=100, type=int, help="Length of the fixed-length genotype")
parser.add_argument("--decay", default=0.99, type=float, help="The decay factor for the epsilon decay (eps_t = eps_0 * decay^t)")
parser.add_argument("--patience", default=50, type=int, help="Number of episodes to use as evaluation period for the early stopping")
parser.add_argument("--timeout", default=600, type=int, help="Maximum evaluation time, useful to continue the evolution in case of MemoryErrors")
parser.add_argument("--with_bias", action="store_true", help="if used, then the the condition will be (sum ...) < <const>, otherwise (sum ...) < 0")
parser.add_argument("--random_init", action="store_true", help="Randomly initializes the leaves in [-1, 1[")
parser.add_argument("--constant_range", default=1000, type=int, help="Max magnitude for the constants being used (multiplied *10^-3). Default: 1000 => constants in [-1, 1]")
parser.add_argument("--constant_step", default=1, type=int, help="Step used to generate the range of constants, mutliplied *10^-3")
parser.add_argument("--types", default=None, type=str, help="This string must contain the range of constants for each variable in the format '#min_0,max_0,step_0,divisor_0;...;min_n,max_n,step_n,divisor_n'. All the numbers must be integers. The min and the max of this range (for each variable) are used to normalize the variables.")


# Setup logging

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = "logs/gym/{}_{}".format(date, "".join(np.random.choice(list(string.ascii_lowercase), size=8)))
logfile = os.path.join(logdir, "log.txt")
os.makedirs(logdir)

args = parser.parse_args()

best = None
lr = "auto" if "." not in args.learning_rate else float(args.learning_rate)

# Creation of the EpsilonDecay Leaf

class EpsilonDecayLeaf(RandomlyInitializedEpsGreedyLeaf):
    """A eps-greedy leaf with epsilon decay."""

    def __init__(self):
        """
        Initializes the leaf
        """
        if not args.random_init:
            RandomlyInitializedEpsGreedyLeaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=0,
                up=0
            )
        else:
            RandomlyInitializedEpsGreedyLeaf.__init__(
                self,
                n_actions=args.n_actions,
                learning_rate=lr,
                discount_factor=args.df,
                epsilon=args.eps,
                low=-1,
                up=1
            )

        self._decay = args.decay
        self._steps = 0

    def get_action(self):
        self.epsilon = self.epsilon * self._decay
        self._steps += 1
        return super().get_action()

# Setup Grammar

input_space_size = args.input_space

types = args.types if args.types is not None else ";".join(["0,10,1,10" for _ in range(input_space_size)])
types = types.replace("#", "")
assert len(types.split(";")) == input_space_size, "Expected {} types, got {}.".format(input_space_size, len(types.split(";")))

consts = {}
for index, type_ in enumerate(types.split(";")):
    rng = type_.split(",")
    start, stop, step, divisor = map(int, rng)
    consts_ = list(map(str, [float(c) / divisor for c in range(start, stop, step)]))
    consts[index] = (consts_[0], consts_[-1])

oblique_split = "+".join(["<const> * (_in_{0} - {1})/({2} - {1})".format(i, consts[i][0], consts[i][1]) for i in range(input_space_size)])

grammar = {
    "bt": ["<if>"],
    "if": ["if <condition>:{<action>}else:{<action>}"],
    "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
    # "const": ["0", "<nz_const>"],
    "const": [str(k/1000) for k in range(-args.constant_range,args.constant_range+1,args.constant_step)]
}

if not args.with_bias:
    grammar["condition"] = [oblique_split + " < 0"]
else:
    grammar["condition"] = [oblique_split + " < <const>"]

print(grammar)

# Seeding of the random number generators

random.seed(args.seed)
np.random.seed(args.seed)


# Log variables

with open(logfile, "a") as f:
    vars_ = locals().copy()
    for k, v in vars_.items():
        f.write("{}: {}\n".format(k, v))


# Definition of the fitness function

def evaluate_fitness(fitness_function, leaf, genotype, episodes=args.episodes):
    repeatable_random_seed = sum(genotype) % (2 ** 31)
    random.seed(args.seed + repeatable_random_seed)
    np.random.seed(args.seed + repeatable_random_seed)
    phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(genotype)
    bt = PythonDT(phenotype, leaf)
    return fitness_function(bt, episodes, timeout=args.timeout)


@stopit.threading_timeoutable(default=((-1000,),None))
def fitness(x, episodes=args.episodes):
    random.seed(args.seed)
    np.random.seed(args.seed)
    global_cumulative_rewards = []
    e = gym.make(args.environment_name)
    initial_perf = None
    try:
        for iteration in range(episodes):
            e.seed(iteration)
            obs = e.reset()
            x.new_episode()
            cum_rew = 0
            action = 0
            previous = None

            for t in range(args.episode_len):
                obs = list(obs.flatten())

                action = x(obs)

                previous = obs[:]

                obs, rew, done, info = e.step(action)

                x.set_reward(rew)

                cum_rew += rew

                if done:
                    break

            x(obs)
            global_cumulative_rewards.append(cum_rew)

            # Check stopping criterion

            if initial_perf is None and iteration >= args.patience:
                initial_perf = np.mean(global_cumulative_rewards)
            elif iteration % args.patience == 0 and iteration > args.patience:
                if np.mean(global_cumulative_rewards[-args.patience:]) - initial_perf < 0:
                    break
                initial_perf = np.mean(global_cumulative_rewards[-args.patience:])
    except Exception as ex:
        if len(global_cumulative_rewards) == 0:
            global_cumulative_rewards = [-1000]
    e.close()

    fitness = np.mean(global_cumulative_rewards[-args.patience:]),
    return fitness, x.leaves


if __name__ == '__main__':
    import collections
    from joblib import parallel_backend

    def fit_fcn(x):
        return evaluate_fitness(fitness, EpsilonDecayLeaf, x)


    with parallel_backend("multiprocessing"):
        pop, log, hof, best_leaves = eval(args.algorithm)(fit_fcn, inputs=input_space_size, leaf=EpsilonDecayLeaf, individuals=args.lambda_, generations=args.generations, jobs=args.jobs, cx_prob=args.cxp, m_prob=args.mp, logfile=logfile, seed=args.seed, mutation=args.mutation, crossover=args.crossover, initial_len=args.genotype_len, selection=args.selection, timeout=None)

        with open(logfile, "a") as log_:
            phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(hof[0])
            phenotype = phenotype.replace('leaf="_leaf"', '')

            for k in range(50000):  # Iterate over all possible leaves
                key = "leaf_{}".format(k)
                if key in best_leaves:
                    v = best_leaves[key].q
                    phenotype = phenotype.replace("out=_leaf", "out={}".format(np.argmax(v)), 1)
                else:
                    break

            log_.write(str(log) + "\n")
            log_.write(str(hof[0]) + "\n")
            log_.write(phenotype + "\n")
            log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))
        with open(os.path.join(logdir, "fitness.tsv"), "w") as f:
            f.write(str(log))

