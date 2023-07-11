#!/usr/bin/env python3.7
import numpy as np
import time
import math
from common import get_args, experiment_setup

if __name__ == '__main__':

    args = get_args()
    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

    if args.learn == "dt-her":
        # generate DT and save all input variables
        list_of_phenotypes, list_of_arm, list_of_goal, list_of_third_coordinate, initial_goals, \
            list_of_goal_first_part, list_of_arm_first_part, list_of_phenotypes_first_part, list_of_arm_second_part, \
            list_of_goal_second_part, list_of_phenotypes_second_part = learner.get_phenotype(args)

        decision_tree_file = open("log/" + args.log_subfolder_name + "/DT.txt", "w")

        if not list_of_phenotypes:
            decision_tree_file.writelines("--------------- Beginning of first part ---------------")
            for number_e, current_dt in enumerate(list_of_phenotypes_first_part):
                header_text_dt = f"--------------- Decision tree{number_e}--------------\n"
                decision_tree_file.writelines(header_text_dt + current_dt)
            decision_tree_file.writelines("--------------- End of first part ---------------")
            decision_tree_file.writelines("--------------- Beginning of second part ---------------")
            for number_e, current_dt in enumerate(list_of_phenotypes_second_part):
                header_text_dt = f"--------------- Decision tree{number_e}--------------\n"
                decision_tree_file.writelines(header_text_dt + current_dt)
            decision_tree_file.writelines("--------------- End of second part ---------------")

        else:
            for number_e, current_dt in enumerate(list_of_phenotypes):
                header_text_dt = f"--------------- Decision tree{number_e}--------------\n"
                decision_tree_file.writelines(header_text_dt + current_dt)

        decision_tree_file.close()

#         list_of_phenotypes_first_part = []
#         for i in range(args.episodes):
#             list_of_phenotypes_first_part.append(""" """)
#
#         list_of_phenotypes_second_part = []
#         for i in range(args.episodes):
#             list_of_phenotypes_second_part.append(""" """)
#
#         list1 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
#                  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, ]
#         list2 = [1, 11, 36, 40, 48]
#         list3 = [49]
#
#         for i in list1:
#             list_of_phenotypes_first_part[i] = """if _in_0 > 2.0:
#     if _in_1 > 17.0:
#         out=2
#
#     else:
#         out=1
#
#
# else:
#     out=3"""
#
#         for i in list2:
#             list_of_phenotypes_first_part[i] = """if _in_0 < 13.0:
#     if _in_0 < 18.0:
#         out=0
#
#     else:
#         if _in_1 < 19.0:
#             out=1
#
#         else:
#             if _in_1 > 5.0:
#                 if _in_0 < 17.0:
#                     if _in_1 < 11.0:
#                         out=2
#
#                     else:
#                         out=3
#
#
#                 else:
#                     if _in_0 > 5.0:
#                         out=3
#
#                     else:
#                         out=3
#
#
#
#             else:
#                 out=1
#
#
#
#
# else:
#     out=1"""
#
#         for i in list3:
#             list_of_phenotypes_first_part[i] = """if _in_0 > 11.0:
#     if _in_1 < 17.0:
#         if _in_1 > 7.0:
#             out=1
#
#         else:
#             if _in_0 > 17.0:
#                 out=3
#
#             else:
#                 if _in_0 > 4.0:
#                     out=2
#
#                 else:
#                     if _in_0 > 1.0:
#                         out=1
#
#                     else:
#                         out=3
#
#
#
#
#
#     else:
#         out=2
#
#
# else:
#     if _in_0 > 11.0:
#         out=3
#
#     else:
#         if _in_1 < 3.0:
#             out=3
#
#         else:
#             if _in_1 > 8.0:
#                 out=0
#
#             else:
#                 if _in_1 > 14.0:
#                     out=0
#
#                 else:
#                     out=1"""
#
#         for i in range(args.episodes):
#             list_of_phenotypes_second_part[i] = """if _in_1 > 4.0:
#     if _in_2 < 11.0:
#         out=4
#
#     else:
#         out=5
#
#
# else:
#     if _in_2 > 19.0:
#         if _in_0 < 4.0:
#             out=0
#
#         else:
#             out=5
#
#
#     else:
#         out=4"""

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



        args.logger.summary_setup()

        for epoch in range(args.epochs):
            for cycle in range(args.cycles):
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()

                list_of_current_arm_position = learner.learn(args, env, env_test, agent, buffer,
                                                             list_of_phenotypes, list_of_arm, list_of_goal,
                                                             list_of_third_coordinate, list_of_current_arm_position,
                                                             list_of_goal_first_part, list_of_arm_first_part,
                                                             list_of_phenotypes_first_part, list_of_arm_second_part,
                                                             list_of_goal_second_part, list_of_phenotypes_second_part)

                tester.cycle_summary()

                # plot
                # np.save('container/initial_goals_ep_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                #         learner.initial_goals_tmp)
                # np.save('container/desired_goals_ep_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                #         learner.desired_goals_tmp)
                # np.save('container/pool_goals_ep_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                #         learner.sampler.pool)
                # np.save('container/achieved_trajectories_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                #         learner.achieved_trajectories_by_robot)
                # np.save('container/achieved_init_states_' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                #         learner.achieved_init_state_by_robot)
                # np.save('container/diffusion_model/d_goal' + str(epoch) + '_cycle' + str(cycle) + '.npy',
                #         learner.diffusion_goal)

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
