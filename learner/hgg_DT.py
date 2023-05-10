import math

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

    def get_intermediate_goal(self, phenotype, current_arm_position, num_dim, third_coordinate):
        # --------------------------------------------------------------------------------
        """
        Arm position:
        [14, 6, 4]
        Goal:
        [14, 9, 4]
        """

        # compute intermediate goal
        # shift every line in phenotype by 4 spaces (1 indent)

        updated_phenotype_with_indents = ""
        for line in phenotype.split('\n'):
            updated_phenotype_with_indents = updated_phenotype_with_indents + "    " + line + "\n"
        # input current position to get next action
        action = self.get_next_action(updated_phenotype_with_indents, current_arm_position)
        # credit to https://stackoverflow.com/questions/33409207/how-to-return-value-from-exec-in-function

        # print("Action: " + str(action))

        next_intermediate_goal = current_arm_position


        # apply action
        if action == 0:
            next_intermediate_goal[0] = next_intermediate_goal[0] + 1
        if action == 1:
            next_intermediate_goal[1] = next_intermediate_goal[1] + 1
        if action == 2:
            next_intermediate_goal[0] = next_intermediate_goal[0] - 1
        if action == 3:
            next_intermediate_goal[1] = next_intermediate_goal[1] - 1

        if num_dim == 3:
            if action == 4:
                next_intermediate_goal[2] = next_intermediate_goal[2] + 1
            if action == 5:
                next_intermediate_goal[2] = next_intermediate_goal[2] - 1

        # print("Intermediate goal before downscale:")
        # print(next_intermediate_goal)

        # Downscale
        next_intermediate_goal = np.array(next_intermediate_goal) / 10
        # Append third coordinate for FetchPush
        if num_dim == 2 and third_coordinate is None:
            raise ValueError("num_dim == 2 but no 3rd coordinate given")

        if num_dim == 2:
            list = next_intermediate_goal.tolist()
            list.append(float(f'{third_coordinate:.1f}'))
            next_intermediate_goal = np.array(list)

        # print("Intermediate goal after downscale:")
        # print(next_intermediate_goal)

        return next_intermediate_goal

    def get_phenotype(self, args, num_dim):
        initial_goals = []
        desired_goals = []
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        print("Vanilla arm pos: ")
        print(str(initial_goals[0]))
        print("Vanilla goal pos: " )
        print(str(desired_goals[0]))
        # TODO: work with all goals? why only with [0]?
        # Compute current arm and goal position
        # Upscale and ceil, because values are in float
        upscaled_arm_position = []
        upscaled_goal = []
        third_coordinate = None
        if num_dim == 3:
            for i in range(len(initial_goals[0])):
                # TODO: trunc or round or ceil?
                upscaled_arm_position.append(math.ceil(initial_goals[0][i] * 10))
                upscaled_goal.append(math.ceil(desired_goals[0][i] * 10))
        if num_dim == 2:
            for i in range(len(initial_goals[0][:2])): # crop to 2 dimensions
                upscaled_arm_position.append(math.ceil(initial_goals[0][i] * 10))
                upscaled_goal.append(math.ceil(desired_goals[0][i] * 10))
            third_coordinate = initial_goals[0][2]

        print("Upscaled Arm position: ")
        print(upscaled_arm_position)
        print("Upscaled Goal: ")
        print(upscaled_goal)
        print("Number of dimensions: " + str(num_dim))

        # generate current DT only once
        # TODO: increase precision by upscaling * 100?
        phenotype = dt.main(grid_size=20, agent_start=upscaled_arm_position, agent_goal=upscaled_goal,
                                  dimensions=num_dim,
                                  reward_type="dense")
        return phenotype, upscaled_arm_position, upscaled_goal, third_coordinate

    def learn(self, args, env, env_test, agent, buffer, phenotype, upscaled_arm_position, upscaled_goal, num_dim, third_coordinate):
        initial_goals = []
        desired_goals = []
        intermediate_goal = []
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        self.sampler.update(initial_goals, desired_goals)

        achieved_trajectories = []
        achieved_init_states = []
        for i in range(args.episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()
            explore_goal = self.sampler.sample(i)

            # self.env_List[i].goal = explore_goal.copy()
            # pass dt and current arm position to get next intermediate goal
            intermediate_goal = np.array(self.get_intermediate_goal(phenotype, upscaled_arm_position, num_dim, third_coordinate))
            self.env_List[i].goal = intermediate_goal


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
        self.sampler.pool = intermediate_goal

        selection_trajectory_idx = {}
        for i in range(self.args.episodes):
            if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())