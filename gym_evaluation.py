#!/usr/bin/python3
"""
In this file one can evaluate programs on OpenAI Gym tasks

Author: Leonardo Lucio Custode
"""
import gym
import cv2
import numpy as np
from time import time, sleep
from matplotlib import pyplot as plt

initial_seed = 200
n_runs = 100
previous_action = False
previous_input = False
time_ = False


def program(input_, step):
    # Insert program here
    if 0.401 * (input_[0] )+-0.104 * (input_[1] )+0.495 * (input_[2] )+-0.055 * (input_[3] )+-0.69 * (input_[4] )+-0.845 * (input_[5] )+-0.2 * (input_[6] )+-0.597 * (input_[7] ) < 0:
        if 0.448 * (input_[0] )+-0.366 * (input_[1] )+0.431 * (input_[2] )+-0.462 * (input_[3] )+-0.693 * (input_[4] )+-0.821 * (input_[5] )+0.461 * (input_[6] )+-0.132 * (input_[7] ) < 0:
            out=3
        else:
            if -0.101 * (input_[0] )+0.133 * (input_[1] )+-0.791 * (input_[2] )+0.653 * (input_[3] )+-0.207 * (input_[4] )+0.731 * (input_[5] )+0.068 * (input_[6] )+0.525 * (input_[7] ) < 0:
                out=2
            else:
                if 0.12 * (input_[0] )+-0.044 * (input_[1] )+-0.772 * (input_[2] )+-0.136 * (input_[3] )+-0.169 * (input_[4] )+0.821 * (input_[5] )+-0.573 * (input_[6] )+-0.251 * (input_[7] ) < 0:
                    out=0
                else:
                    out=2
    else:
        out=1
    # End program
    return out


def evaluate(program, seed):
    env = gym.make("LunarLander-v2")
    # env._max_episode_steps = 10000
    env.seed(seed)
    obs = env.reset()

    score = 0
    i = 0
    action = 0
    previous_state = obs[:]

    t = time()
    while True:
        input_ = []
        obs = list(obs)

        if previous_input:
            input_.extend(previous_state)
        if time_:
            input_.append(i)
        if previous_action:
            input_.append(action)
        input_.extend(obs)

        output = program(input_, i)
        i += 1
        action = output
        previous_state = obs[:]

        obs, rew, done, info = env.step(output)
        score += rew

        if done:
            break
    env.close()

    cv2.destroyAllWindows()
    print("Evaluation took {}s".format(time()-t))
    print("Score: {}".format(score))
    return score


scores = [evaluate(program, s) for s in range(initial_seed, initial_seed + n_runs)]
print("Mean: {}; Std: {}".format(np.mean(scores), np.std(scores)))
