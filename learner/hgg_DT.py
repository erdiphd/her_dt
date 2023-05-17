import math
import random
import numpy as np
from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
from ge_q_dts.dt import EpsGreedyLeaf, PythonDT
from ge_q_dts import simple_test_orthogonal as dt
import ast
import copy


class TrajectoryPool_DT:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])


class MatchSampler_DT:
    def __init__(self, args, achieved_trajectory_pool):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)
        self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
        self.delta = self.env.distance_threshold
        self.goal_distance = get_goal_distance(args)

        self.length = args.episodes
        init_goal = self.env.reset()['achieved_goal'].copy()
        self.pool = np.tile(init_goal[np.newaxis, :], [self.length, 1]) + np.random.normal(0, self.delta,
                                                                                           size=(self.length, self.dim))
        self.init_state = self.env.reset()['observation'].copy()

        self.match_lib = gcc_load_lib('learner/cost_flow.c')
        self.achieved_trajectory_pool = achieved_trajectory_pool

        # estimating diameter
        self.max_dis = 0
        for i in range(1000):
            obs = self.env.reset()
            dis = self.goal_distance(obs['achieved_goal'], obs['desired_goal'])
            if dis > self.max_dis: self.max_dis = dis

    def add_noise(self, pre_goal, noise_std=None):
        goal = pre_goal.copy()
        dim = 2 if self.args.env[:5] == 'Fetch' else self.dim
        if noise_std is None: noise_std = self.delta
        goal[:dim] += np.random.normal(0, noise_std, size=dim)
        return goal.copy()

    def sample(self, idx):
        if self.args.env[:5] == 'Fetch':
            return self.add_noise(self.pool[idx])
        else:
            return self.pool[idx].copy()

    def find(self, goal):
        res = np.sqrt(np.sum(np.square(self.pool - goal), axis=1))
        idx = np.argmin(res)
        if test_pool:
            self.args.logger.add_record('Distance/sampler', res[idx])
        return self.pool[idx].copy()

    def update(self, initial_goals, desired_goals):
        if self.achieved_trajectory_pool.counter == 0:
            self.pool = copy.deepcopy(desired_goals)
            return

        achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
        candidate_goals = []
        candidate_edges = []
        candidate_id = []

        agent = self.args.agent
        achieved_value = []
        for i in range(len(achieved_pool)):
            obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
                   range(achieved_pool[i].shape[0])]
            feed_dict = {
                agent.raw_obs_ph: obs
            }
            value = agent.sess.run(agent.q_pi, feed_dict)[:, 0]
            value = np.clip(value, -1.0 / (1.0 - self.args.gamma), 0)
            achieved_value.append(value.copy())

        n = 0
        graph_id = {'achieved': [], 'desired': []}
        for i in range(len(achieved_pool)):
            n += 1
            graph_id['achieved'].append(n)
        for i in range(len(desired_goals)):
            n += 1
            graph_id['desired'].append(n)
        n += 1
        self.match_lib.clear(n)

        for i in range(len(achieved_pool)):
            self.match_lib.add(0, graph_id['achieved'][i], 1, 0)

        for i in range(len(achieved_pool)):
            for j in range(len(desired_goals)):
                res = np.sqrt(np.sum(np.square(achieved_pool[i] - desired_goals[j]), axis=1)) - achieved_value[i] / (
                        self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
                match_dis = np.min(res) + self.goal_distance(achieved_pool[i][0], initial_goals[j]) * self.args.hgg_c
                match_idx = np.argmin(res)

                edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
                candidate_goals.append(achieved_pool[i][match_idx])
                candidate_edges.append(edge)
                candidate_id.append(j)

        for i in range(len(desired_goals)):
            self.match_lib.add(graph_id['desired'][i], n, 1, 0)

        match_count = self.match_lib.cost_flow(0, n)
        print("Match_count: " + str(match_count))
        print("length : " + str(self.length))
        assert match_count == self.length

        explore_goals = [0] * self.length
        for i in range(len(candidate_goals)):
            if self.match_lib.check_match(candidate_edges[i]) == 1:
                explore_goals[candidate_id[i]] = candidate_goals[i].copy()
        assert len(explore_goals) == self.length
        self.pool = np.array(explore_goals)


class HGGLearner_DT:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)
        self.goal_distance = get_goal_distance(args)

        # save arrays
        self.initial_goals_tmp = []
        self.desired_goals_tmp = []
        self.achieved_trajectories_by_robot = []
        self.achieved_init_state_by_robot = []
        self.action_list = []
        self.diffusion_goal = []

        self.env_List = []
        for i in range(args.episodes):
            self.env_List.append(make_env(args))

        self.achieved_trajectory_pool = TrajectoryPool_DT(args, args.hgg_pool_size)
        self.sampler = MatchSampler_DT(args, self.achieved_trajectory_pool)

    def convertExpr2Expression(self, Expr):
        Expr.lineno = 0
        Expr.col_offset = 0
        result = ast.Expression(Expr.value, lineno=0, col_offset=0)

        return result

    def exec_with_return(self, code, variables):
        code_ast = ast.parse(code)

        init_ast = copy.deepcopy(code_ast)
        init_ast.body = code_ast.body[:-1]

        last_ast = copy.deepcopy(code_ast)
        last_ast.body = code_ast.body[-1:]

        exec(compile(init_ast, "<ast>", "exec"), variables)
        if type(last_ast.body[0]) == ast.Expr:
            return eval(compile(self.convertExpr2Expression(last_ast.body[0]), "<ast>", "eval"), variables)
        else:
            exec(compile(last_ast, "<ast>", "exec"), variables)

    def get_next_action(self, phenotype, input):
        variables = {}  # {"out": None, "leaf": None}
        for idx, i in enumerate(input):
            variables["_in_{}".format(idx)] = i

        # print(variables)
        # position = [0, 0, 0]
        # variables = {'_in_0' : 0, '_in_1' : 2, '_in_2' : 0}
        # print("def func(): \n" + phenotype + "\n    return out \nfunc()")

        return self.exec_with_return("def func(): \n" + phenotype + "    return out \nfunc()", variables)

    def get_intermediate_goal(self, args, phenotype, current_arm_position, num_dim, third_coordinate,
                              append_3rd_coordinate):

        # compute intermediate goal

        """
        Arm position:
        [1.5, 0.6, 0.4]
        Goal:
        [1.5, 0.9, 0.4]
        """
        # pass to DT upscaled arm position, because it only works with upscaled
        upscaled_arm_position = np.array(current_arm_position) * 10
        # pretending than the coordinate is 8 instead of 8.25. This is needed for the 0.025 step size
        for i in range(len(upscaled_arm_position)):
            upscaled_arm_position[i] = math.trunc(upscaled_arm_position[i])

        action = 0
        # shift every line in phenotype by 4 spaces (1 indent)
        updated_phenotype_with_indents = ""
        for line in phenotype.split('\n'):
            updated_phenotype_with_indents = updated_phenotype_with_indents + "    " + line + "\n"
        # intermediate goal is always 3D but DT is optimized for 2D (FetchPush) => crop current position
        if num_dim == 2 and len(upscaled_arm_position) == 3:
            # input current position to get next action
            action = self.get_next_action(updated_phenotype_with_indents, np.copy(upscaled_arm_position[:2]))
        else:
            action = self.get_next_action(updated_phenotype_with_indents, np.copy(upscaled_arm_position))
        # credit to https://stackoverflow.com/questions/33409207/how-to-return-value-from-exec-in-function

        # print("Action: " + str(action))

        next_intermediate_goal = current_arm_position.copy()

        # apply action. Only DT is working with 10X upscaling and actions with "1" steps.
        # No need to upscale and downscale every goal in HGG.
        # When applying, action step will just be 0.1 instead of 1
        # Dynamic step size: 0.1 -> 2 goals, 0.025 -> 8, 0.0005 -> 400
        if action == 0:
            next_intermediate_goal[0] = next_intermediate_goal[0] + args.hgg_dt_step_size
        if action == 1:
            next_intermediate_goal[1] = next_intermediate_goal[1] + args.hgg_dt_step_size
        if action == 2:
            next_intermediate_goal[0] = next_intermediate_goal[0] - args.hgg_dt_step_size
        if action == 3:
            next_intermediate_goal[1] = next_intermediate_goal[1] - args.hgg_dt_step_size

        if num_dim == 3:
            if action == 4:
                next_intermediate_goal[2] = next_intermediate_goal[2] + args.hgg_dt_step_size
            if action == 5:
                next_intermediate_goal[2] = next_intermediate_goal[2] - args.hgg_dt_step_size

        # Append third coordinate for FetchPush, because it stays the same
        if num_dim == 2 and third_coordinate is None:
            raise ValueError("num_dim == 2 but no 3rd coordinate given")

        if append_3rd_coordinate is True:
            if num_dim == 2:
                temp_list = next_intermediate_goal.copy()
                temp_list.append(float(f'{third_coordinate:.2f}'))
                next_intermediate_goal = np.array(temp_list)

        # print("Intermediate goal:")
        # print(next_intermediate_goal)

        return next_intermediate_goal.copy()

    def get_phenotype(self, args, num_dim):
        initial_goals = []
        desired_goals = []
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        # Compute current arm and goal position
        # Upscale and ceil, because values are in float

        list_of_phenotypes = []
        list_of_arm = []
        list_of_goal = []
        list_of_third_coordinate = []
        third_coordinate = None
        # Process all start-goal pairs -> generate DT for each
        for j in range(args.episodes):
            # rewrite coordinates and put them into big list
            upscaled_arm_position = []
            upscaled_goal = []
            third_coordinate = None
            if num_dim == 3:
                for i in range(len(initial_goals[j])):
                    # TODO: trunc or round or ceil?
                    # use 10 X upscaling
                    upscaled_arm_position.append(math.ceil(initial_goals[j][i] * 10))
                    upscaled_goal.append(math.ceil(desired_goals[j][i] * 10))
            if num_dim == 2:
                for i in range(len(initial_goals[j][:2])):  # crop to 2 dimensions for FetchPush
                    upscaled_arm_position.append(math.ceil(initial_goals[j][i] * 10))
                    upscaled_goal.append(math.ceil(desired_goals[j][i] * 10))
                third_coordinate = initial_goals[j][2]

            # print("Upscaled Arm position: ")
            # print(upscaled_arm_position)
            # print("Upscaled Goal: ")
            # print(upscaled_goal)
            # print("Number of dimensions: " + str(num_dim))

            # generate current DT only once for every start-goal pair
            phenotype = dt.main(grid_size=20, agent_start=upscaled_arm_position, agent_goal=upscaled_goal,
                                dimensions=num_dim,
                                reward_type="dense")
            print("Phenotype number " + str(j) + " generated")
            list_of_phenotypes.append(phenotype)
            list_of_arm.append(upscaled_arm_position)
            list_of_goal.append(upscaled_goal)
            list_of_third_coordinate.append(third_coordinate)
        return list_of_phenotypes, list_of_arm, list_of_goal, list_of_third_coordinate, initial_goals

    def learn(self, args, env, env_test, agent, buffer, list_of_phenotypes, list_of_arm, list_of_goal, num_dim,
              list_of_third_coordinate, list_of_current_arm_position, append_3rd_coordinate):
        initial_goals = []
        desired_goals = []
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        self.sampler.update(initial_goals, desired_goals)

        achieved_trajectories = []
        achieved_init_states = []
        pre_goal = []

        for i in range(args.episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()
            explore_goal = self.sampler.sample(i)
            intermediate_goal = []
            # create a goal to compare intermediate goal to
            current_goal = list_of_goal[i].copy()
            current_goal = np.array(current_goal) / 10
            current_goal = current_goal.tolist()
            if num_dim == 2:
                current_goal.append(float(f'{list_of_third_coordinate[i]:.2f}'))

            goal_reached = False
            # preparation for equality check
            tmp_arm = []
            tmp_goal = []
            for j in range(len(list_of_current_arm_position[i])):
                # round, to prevent phantom decimal points like 0.7000000000000001
                tmp_arm.append(round(list_of_current_arm_position[i][j], 10))
                # here: 0.899999999999999
                tmp_goal.append(round(current_goal[j], 2))

            # print("Temp arm: ")
            # print(tmp_arm)
            # print("Temp goal: ")
            # print(tmp_goal)

            # check point so the intermediate goal won't run away from the desired goal
            if np.array_equal(tmp_goal, tmp_arm):
                goal_reached = True
                intermediate_goal = list_of_current_arm_position[i].copy()

            if goal_reached is False:
                # pass dt and current arm position to get next intermediate goal
                # return 1 intermediate goal for every start-goal pair
                intermediate_goal = np.array(self.get_intermediate_goal(args, list_of_phenotypes[i],
                                                                        list_of_current_arm_position[i].copy(), num_dim,
                                                                        list_of_third_coordinate[i].copy(),
                                                                        append_3rd_coordinate))
            pre_goal.append(intermediate_goal)
        # interrupt loop to add noise to pre_goal x-coordinate. Dont touch y-coordinate, its moving in DT
        if args.env == "FetchPush-v1":
            for i in range(args.episodes):
                pre_goal[i][0] = round(random.uniform((pre_goal[i][0]) - 0.05, (pre_goal[i][0]) + 0.05), 2)
                # pre_goal[i][1] = round(random.uniform((pre_goal[i][1]), (pre_goal[i][1]) + 0.09), 2)
                # pre_goal[i][2] = round(random.uniform((pre_goal[i][2]), (pre_goal[i][2]) + 0.09), 2)

        # continue with noised goals
        for i in range(args.episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()
            explore_goal = self.sampler.sample(i)

            self.env_List[i].goal = pre_goal[i].copy()
            # make the intermediate goal move
            list_of_current_arm_position[i] = pre_goal[i].copy()

            # print("Upscaled arm position: ")
            # print(list_of_current_arm_position[i])
            # print("Intermediate goal: ")
            # print(intermediate_goal)
            # print("Desired goal: ")
            # print(list_of_goal[i])

            obs = self.env_List[i].get_obs()
            current = Trajectory(obs)
            trajectory = [obs['achieved_goal'].copy()]
            for timestep in range(args.timesteps):
                action = agent.step(obs, explore=True)
                obs, reward, done, info = self.env_List[i].step(action)
                trajectory.append(obs['achieved_goal'].copy())
                if timestep == args.timesteps - 1: done = True
                current.store_step(action, obs, reward, done)
                if done: break
            achieved_trajectories.append(np.array(trajectory))
            achieved_init_states.append(init_state)
            buffer.store_trajectory(current)
            agent.normalizer_update(buffer.sample_batch())

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                agent.target_update()

        # edit for plotting
        self.achieved_trajectories_by_robot = achieved_trajectories
        self.achieved_init_state_by_robot = achieved_init_states
        self.initial_goals_tmp = initial_goals
        self.desired_goals_tmp = desired_goals
        # pass list of intermediate goals
        self.sampler.pool = list_of_current_arm_position
        # print("List of current arm: ")
        # print(list_of_current_arm_position)

        selection_trajectory_idx = {}
        for i in range(self.args.episodes):
            if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())

        return list_of_current_arm_position
