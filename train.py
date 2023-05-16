#!/usr/bin/env python3.7
import numpy as np
import time
import math
from common import get_args,experiment_setup

if __name__ == '__main__':

	args = get_args()
	env, env_test, agent, buffer, learner, tester = experiment_setup(args)
	# generate DT only once
	num_dim = 0
	if args.env == "FetchPush-v1":
		num_dim = 2 # adjust --input_space, n_actions, types arguments in simple_test
	else:
		num_dim = 3
	list_of_phenotypes, list_of_arm, list_of_goal, list_of_third_coordinate, initial_goals = learner.get_phenotype(args, num_dim)

	# log phenotypes for debugging
	# for i in range(len(list_of_phenotypes)):
	# 	with open('log/phenotypes/HGG_phenotype_2.txt', 'w') as f:
	# 		f.write(str(list_of_phenotypes[i]) + "\n" + "-------------------------")

	# print(list_of_phenotypes)

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
	list_of_current_arm_position = []
	if num_dim == 2:
		for i in range(args.episodes):
			current_arm_position = []
			for j in range(len(initial_goals[i][:2])):
				current_arm_position.append(math.ceil(initial_goals[i][j] * 10) / 10)
			list_of_current_arm_position.append(current_arm_position)
	if num_dim == 3:
		for i in range(args.episodes):
			current_arm_position = []
			for j in range(len(initial_goals[i])):
				current_arm_position.append(math.ceil(initial_goals[i][j] * 10) / 10)
			list_of_current_arm_position.append(current_arm_position)
	append_3rd_coordinate = True

# 	list_of_phenotypes = []
# 	for i in range(args.episodes):
# 		list_of_phenotypes.append("""if _in_1 > 13.0:
#     if _in_0 > 3.0:
#         if _in_0 < 3.0:
#             if _in_1 < 6.0:
#                 out=0
#
#             else:
#                 if _in_1 < 7.0:
#                     out=1
#
#                 else:
#                     out=3
#
#
#
#         else:
#             out=0
#
#
#     else:
#         out=3
#
#
# else:
#     if _in_0 < 12.0:
#         out=2
#
#     else:
#         if _in_1 > 8.0:
#             out=2
#
#         else:
#             out=1""")

	# coordinates are not upscaled. Only DT is working with upscaled coordinates
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
			list_of_current_arm_position = learner.learn(args, env, env_test, agent, buffer, list_of_phenotypes, list_of_arm, list_of_goal, num_dim, list_of_third_coordinate, list_of_current_arm_position, append_3rd_coordinate)
			tester.cycle_summary()
			# append 3rd coordinate only once at first run
			append_3rd_coordinate = False

			# plot
			np.save('container/initial_goals_ep_'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.initial_goals_tmp)
			np.save('container/desired_goals_ep_'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.desired_goals_tmp)
			# TODO: check, changed saving here
			np.save('container/pool_goals_ep_'+str(epoch) + '_cycle' + str(cycle) + '.npy', list_of_current_arm_position)
			np.save('container/achieved_trajectories_'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.achieved_trajectories_by_robot)
			np.save('container/achieved_init_states_'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.achieved_init_state_by_robot)
			np.save('container/diffusion_model/d_goal'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.diffusion_goal)


			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', buffer.steps_counter)
			args.logger.add_record('TimeCost(sec)', time.time()-start_time)

			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)

		tester.epoch_summary()

	tester.final_summary()
