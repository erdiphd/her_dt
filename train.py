#!/usr/bin/env python3.7
import numpy as np
import time
import math
from common import get_args, experiment_setup

if __name__ == '__main__':

    args = get_args()
    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

    if args.learn == "hgg_dt":
        # generate DT only once
        list_of_phenotypes, list_of_arm, list_of_goal, list_of_third_coordinate, initial_goals, list_of_goal_first_part, list_of_arm_first_part, list_of_phenotypes_first_part, list_of_arm_second_part, list_of_goal_second_part, list_of_phenotypes_second_part = learner.get_phenotype(args)

        args.logger.summary_init(agent.graph, agent.sess)

        # Progress info
        args.logger.add_item('Epoch')
        args.logger.add_item('Cycle')
        args.logger.add_item('Episodes@green')
        args.logger.add_item('Timesteps')
        args.logger.add_item('TimeCost(sec)')

        # Algorithm info
        for key in agent.train_info.keys():
            args.logger.add_item(key, 'scalar')

        # Test info
        for key in tester.info:
            args.logger.add_item(key, 'scalar')

        # at first initialize current position with initial_goal for all start-goal pairs
        # coordinates are not upscaled. Only DT is working with upscaled coordinates
        list_of_current_arm_position = []
        for i in range(args.episodes):
            current_arm_position = []
            for j in range(2):
                current_arm_position.append(math.ceil(initial_goals[i][j] * 10) / 10)
            current_arm_position.append(initial_goals[i][2])
            list_of_current_arm_position.append(current_arm_position)

        list_of_phenotypes_first_part = []
        for i in range(args.episodes):
            list_of_phenotypes_first_part.append(""" """)

        for i in [0, 1, 5, 7, 8, 9, 10, 22, 26, 28, 30, 33, 34, 36, 38, 42]:
            list_of_phenotypes_first_part[i] = """if _in_1 < 9.0:
    out=1

else:
    out=1"""

        for i in [2, 4, 18, 27, 29, 39, 47, 49]:
            list_of_phenotypes_first_part[i] = """if _in_0 < 14.0:
    out=0

else:
    out=1"""

        for i in [3, 6, 13, 15, 24, 46]:
            list_of_phenotypes_first_part[i] = """if _in_0 > 13.0:
    out=2

else:
    out=1"""

        for i in [11, 17, 20, 35, 40]:
            list_of_phenotypes_first_part[i] = """if _in_0 > 14.0:
    out=2

else:
    out=1"""

        for i in [12, 14, 21, 25, 31, 37, 41, 44, 45, 48]:
            list_of_phenotypes_first_part[i] = """if _in_0 < 15.0:
    out=0

else:
    out=1"""
        for i in [16, 32]:
            list_of_phenotypes_first_part[i] = """if _in_0 > 13.0:
    out=2

else:
    out=1"""
        for i in [19, 43]:
            list_of_phenotypes_first_part[i] = """if _in_0 > 12.0:
    out=2

else:
    out=1"""

        for i in [23]:
            list_of_phenotypes_first_part[i] = """if _in_0 < 15.0:
    out=0

else:
    out=1"""



#         """
#         14, 6
#         14, 9
#         """
#         pos_1 = [0, 1, 5, 7, 8, 9, 10, 22, 26, 28, 30, 33, 34, 36, 38, 42]
#         target_1 = """if _in_1 < 9.0:
#     out=1
#
# else:
#     out=1"""
#
#
#         """
#         13, 6
#         14, 9
#         """
#         pos_2 = [2, 4, 18, 27, 29, 39, 47, 49]
#         target_2 = """if _in_0 < 14.0:
#     out=0
#
# else:
#     out=1"""
#
#         """
#         14, 6
#         13, 9
#         """
#         pos_3 = [3, 6, 13, 15, 24, 46]
#         target_3 = """if _in_0 > 13.0:
#     out=2
#
# else:
#     out=1"""
#
#         """
#         15, 6
#         14, 9
#         """
#         pos_4 = [11, 17, 20, 35, 40]
#         target_4 = """if _in_0 > 14.0:
#     out=2
#
# else:
#     out=1"""
#
#         """
#         13, 6
#         15, 9
#         """
#         pos_5 = [12, 14, 21, 25, 31, 37, 41, 44, 45, 48]
#         target_5 = """if _in_0 < 15.0:
#     out=0
#
# else:
#     out=1"""
#
#         """
#         15, 6
#         13, 9
#         """
#         pos_6 = [16, 32]
#         target_6 = """if _in_0 > 13.0:
#     out=2
#
# else:
#     out=1"""
#
#         """
#         15, 6
#         12, 9
#         """
#         pos_7 = [19, 43]
#         target_7 = """if _in_0 > 12.0:
#     out=2
#
# else:
#     out=1"""
#
#         """
#         12, 6
#         15, 9
#         """
#         pos_8 = [23]
#         target_8 = """if _in_0 < 15.0:
#     out=0
#
# else:
#     out=1"""

        # for x,y in zip(pos_1, target_1):
        #     list_of_phenotypes_first_part[x] = y
        # for x,y in zip(pos_2, target_2):
        #     list_of_phenotypes_first_part[x] = y
        # for x,y in zip(pos_3, target_3):
        #     list_of_phenotypes_first_part[x] = y
        # for x,y in zip(pos_4, target_4):
        #     list_of_phenotypes_first_part[x] = y
        # for x,y in zip(pos_5, target_5):
        #     list_of_phenotypes_first_part[x] = y
        # for x,y in zip(pos_6, target_6):
        #     list_of_phenotypes_first_part[x] = y
        # for x,y in zip(pos_7, target_7):
        #     list_of_phenotypes_first_part[x] = y
        # for x,y in zip(pos_8, target_8):
        #     list_of_phenotypes_first_part[x] = y


#         for i in range(args.episodes):
#             list_of_phenotypes_first_part.append("""if _in_0 > 13.0:
#     out=2
#
# else:
#     out=1""")

        list_of_phenotypes_second_part = []
        for i in range(args.episodes):
            list_of_phenotypes_second_part.append("""if _in_2 < 9.0:
    out=4

else:
    out=2""")

        """
        Arm position:
        [1.5, 0.6, 0.4]
        Goal:
        [1.5, 0.9, 0.4]
        """
        args.logger.summary_setup()

        for epoch in range(args.epochs):
            for cycle in range(args.cycles):
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()

                # learner.learn(args, env, env_test, agent, buffer)
                list_of_current_arm_position = learner.learn(args, env, env_test, agent, buffer,
                                                                                   list_of_phenotypes, list_of_arm,
                                                                                   list_of_goal,
                                                                                   list_of_third_coordinate,
                                                                                   list_of_current_arm_position,
                                                                                   list_of_goal_first_part,
                                                                                   list_of_arm_first_part, list_of_phenotypes_first_part, list_of_arm_second_part, list_of_goal_second_part, list_of_phenotypes_second_part)
                # print("achieved trajectories: ")
                # print(learner.achieved_trajectories_by_robot)
                tester.cycle_summary()

                # plot
                np.save('container/initial_goals_ep_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                        learner.initial_goals_tmp)
                np.save('container/desired_goals_ep_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                        learner.desired_goals_tmp)
                np.save('container/pool_goals_ep_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                        list_of_current_arm_position)
                np.save('container/achieved_trajectories_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                        learner.achieved_trajectories_by_robot)
                np.save('container/achieved_init_states_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                        learner.achieved_init_state_by_robot)
                np.save('container/diffusion_model/d_goal' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                        learner.diffusion_goal)

                args.logger.add_record('Epoch', str(epoch) + '/' + str(args.epochs))
                args.logger.add_record('Cycle', str(cycle) + '/' + str(args.cycles))
                args.logger.add_record('Episodes', buffer.counter)
                args.logger.add_record('Timesteps', buffer.steps_counter)
                args.logger.add_record('TimeCost(sec)', time.time() - start_time)

                args.logger.tabular_show(args.tag)
                args.logger.summary_show(buffer.counter)

            tester.epoch_summary()

        tester.final_summary()
    else:
        args.logger.summary_init(agent.graph, agent.sess)

        # Progress info
        args.logger.add_item('Epoch')
        args.logger.add_item('Cycle')
        args.logger.add_item('Episodes@green')
        args.logger.add_item('Timesteps')
        args.logger.add_item('TimeCost(sec)')

        # Algorithm info
        for key in agent.train_info.keys():
            args.logger.add_item(key, 'scalar')

        # Test info
        for key in tester.info:
            args.logger.add_item(key, 'scalar')

        args.logger.summary_setup()

        for epoch in range(args.epochs):
            for cycle in range(args.cycles):
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()

                learner.learn(args, env, env_test, agent, buffer)
                tester.cycle_summary()

                args.logger.add_record('Epoch', str(epoch) + '/' + str(args.epochs))
                args.logger.add_record('Cycle', str(cycle) + '/' + str(args.cycles))
                args.logger.add_record('Episodes', buffer.counter)
                args.logger.add_record('Timesteps', buffer.steps_counter)
                args.logger.add_record('TimeCost(sec)', time.time() - start_time)

                args.logger.tabular_show(args.tag)
                args.logger.summary_show(buffer.counter)

            tester.epoch_summary()

        tester.final_summary()
