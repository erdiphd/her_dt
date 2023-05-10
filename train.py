#!/usr/bin/env python3.7
import numpy as np
import time
from common import get_args,experiment_setup

if __name__=='__main__':


	args = get_args()
	env, env_test, agent, buffer, learner, tester = experiment_setup(args)
	# generate DT only once
	num_dim = 0
	if args.env == "FetchPush-v1":
		num_dim = 2 # adjust --input_space, n_actions, types arguments in simple_test
	else:
		num_dim = 3
	phenotype, upscaled_arm_position, upscaled_goal, third_coordinate = learner.get_phenotype(args, num_dim)

	# print("Phenotype: ")
	# print(str(phenotype))

	# log phenotype
	# with open('log/DT/phenotype.txt', 'w') as f:
	# 	f.write(str(phenotype))

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

			# learner.learn(args, env, env_test, agent, buffer)
			learner.learn(args, env, env_test, agent, buffer, phenotype, upscaled_arm_position, upscaled_goal, num_dim, third_coordinate)
			tester.cycle_summary()

			# plot
			np.save('container/initial_goals_ep_'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.initial_goals_tmp)
			np.save('container/desired_goals_ep_'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.desired_goals_tmp)
			np.save('container/pool_goals_ep_'+str(epoch) + '_cycle' + str(cycle) + '.npy', learner.sampler.pool)
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
