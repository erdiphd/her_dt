import math
import random
import numpy as np
from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
from ge_q_dts.dt import EpsGreedyLeaf, PythonDT
from ge_q_dts import simple_test_orthogonal as dt
from ge_q_dts import simple_test_orthogonal_2 as dt_2
from ge_q_dts import simple_test_orthogonal_3 as dt_3
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
        self.delta = self.env.distance_threshold
        self.dim = np.prod(self.env.reset()['achieved_goal'].shape)

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

        return self.exec_with_return("def func(): \n" + phenotype + "    return out \nfunc()", variables)

    def get_intermediate_goal(self, args, phenotype, current_arm_position, third_coordinate, feedback):

        # compute intermediate goal

        """
        Arm position:
        [1.5, 0.6, 0.4]
        Goal:
        [1.5, 0.9, 0.4]
        """
        # pass to DT upscaled arm position, because it only works with upscaled.
        # But no downscale, because we use action only as direction, not as step size
        upscaled_arm_position = np.array(current_arm_position) * 10

        # shift every line in phenotype by 4 spaces (1 indent). This is needed to be able to read the DT
        updated_phenotype_with_indents = ""
        for line in phenotype.split('\n'):
            updated_phenotype_with_indents = updated_phenotype_with_indents + "    " + line + "\n"
        # getting action from reading the DT after passing current position (last intermediate goal)
        action = self.get_next_action(updated_phenotype_with_indents, np.copy(upscaled_arm_position))
        # credit to https://stackoverflow.com/questions/33409207/how-to-return-value-from-exec-in-function

        # print("Action: " + str(action))

        next_intermediate_goal = current_arm_position.copy()

        # apply action. Only DT is working with 10X upscaling and actions with "1" steps, the algorithm not.
        # No need to upscale and downscale every goal in HGG, action is only direction info, step size can be of anything
        # Dynamic step size: 0.1 -> 2 goals, 0.025 -> 8, 0.00075 -> 400. Impacted by feedback
        if action == 0:
            next_intermediate_goal[0] = next_intermediate_goal[0] + args.hgg_dt_step_size * feedback
        if action == 1:
            next_intermediate_goal[1] = next_intermediate_goal[1] + args.hgg_dt_step_size * feedback
        if action == 2:
            next_intermediate_goal[0] = next_intermediate_goal[0] - args.hgg_dt_step_size * feedback
        if action == 3:
            next_intermediate_goal[1] = next_intermediate_goal[1] - args.hgg_dt_step_size * feedback
        # action 4 and 5 can only be called with 3D DT
        if action == 4:
            next_intermediate_goal[2] = next_intermediate_goal[2] + args.hgg_dt_step_size * feedback
        if action == 5:
            next_intermediate_goal[2] = next_intermediate_goal[2] - args.hgg_dt_step_size * feedback

        # Append third coordinate for FetchPush and FetchSlide, because it stays the same
        if args.env == "FetchPush-v1" or args.env == "FetchSlide-v1":
            # if third coordinate was not appended yet
            if len(next_intermediate_goal) == 2:
                next_intermediate_goal.append(float(f'{third_coordinate:.5f}'))
            else:
                next_intermediate_goal[2] = float(f'{third_coordinate:.5f}')
        # Append third coordinate for the first part of the FetchPickAndPlace or FetchReach, because first part is only xy
        if third_coordinate is not None:
            if len(next_intermediate_goal) == 2:
                next_intermediate_goal.append(float(f'{third_coordinate:.1f}'))
            # else:
            #     next_intermediate_goal[2] = float(f'{third_coordinate:.1f}')

        return next_intermediate_goal.copy()

    def get_phenotype(self, args):
        initial_goals = []
        desired_goals = []
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        list_of_phenotypes = []
        list_of_arm = []
        list_of_goal = []
        list_of_third_coordinate = []

        list_of_phenotypes_first_part = []
        list_of_arm_first_part = []
        list_of_goal_first_part = []
        list_of_phenotypes_second_part = []
        list_of_arm_second_part = []
        list_of_goal_second_part = []
        if args.env == "FetchPush-v1" or args.env == "FetchSlide-v1":
            # Process all start-goal pairs -> generate DT for each
            for j in range(args.episodes):
                # rewrite coordinates and put them into big list
                upscaled_arm_position = []
                upscaled_goal = []
                # Compute upscaled arm and goal position
                # Upscale and ceil, because values are in float
                for i in range(2):
                    # use 10 X upscaling to work with DT
                    upscaled_arm_position.append(math.ceil(initial_goals[j][i] * 10))
                    upscaled_goal.append(math.ceil(desired_goals[j][i] * 10))
                # save third coordinate as float to append it later
                third_coordinate = initial_goals[j][2]

                # print("Initial goals: ")
                # print(initial_goals[j])
                # print("Desired goals: ")
                # print(desired_goals[j])

                if args.obstacle is False:
                    # generate current DT only once for every start-goal pair
                    # working here with 2D DT, use sparse reward
                    phenotype = dt_2.main(grid_size=20, agent_start=upscaled_arm_position, agent_goal=upscaled_goal,
                                        dimensions=2,
                                        reward_type="sparse", obstacle_is_on=args.obstacle)
                    print("Phenotype number " + str(j) + " generated")
                    list_of_phenotypes.append(phenotype)
                    list_of_arm.append(upscaled_arm_position)
                    list_of_goal.append(upscaled_goal)
                    list_of_third_coordinate.append(third_coordinate)
                elif args.obstacle is True:
                    # if we are working with obstacles, use other file with larger number of generations
                    # + only dense function
                    phenotype = dt_3.main(grid_size=20, agent_start=upscaled_arm_position, agent_goal=upscaled_goal,
                                        dimensions=2,
                                        reward_type="dense", obstacle_is_on=args.obstacle)
                    print("Phenotype number " + str(j) + " generated")
                    list_of_phenotypes.append(phenotype)
                    list_of_arm.append(upscaled_arm_position)
                    list_of_goal.append(upscaled_goal)
                    list_of_third_coordinate.append(third_coordinate)
        else:
            # generate 2 dt for each start-goal pair -> first one with 2 dim, second one only for third dim
            # this is done because DT cannot work with complex 3D problems

            # first part: x,y coordinates
            for j in range(args.episodes):
                # rewrite coordinates and put them into big list
                upscaled_arm_position = []
                upscaled_goal = []
                for i in range(2):
                    # use 10 X upscaling to work with DT
                    upscaled_arm_position.append(math.ceil(initial_goals[j][i] * 10))
                    upscaled_goal.append(math.ceil(desired_goals[j][i] * 10))
                # first DT is 2d, so remember third coordinate to append it later
                third_coordinate = math.ceil(initial_goals[j][2] * 10) / 10

                # print("initial pos:")
                # print(initial_goals[j])
                # print("desired goal: ")
                # print(desired_goals[j])
                # print("First part: arm: ")
                # print(upscaled_arm_position)
                # print("First part: goal: ")
                # print(upscaled_goal)

                # generate current DT only once for every start-goal pair
                # working here with 2D DT, use sparse reward
                phenotype = dt_2.main(grid_size=20, agent_start=upscaled_arm_position, agent_goal=upscaled_goal,
                                    dimensions=2,
                                    reward_type="sparse", obstacle_is_on=False)
                print("Phenotype part 1 number " + str(j) + " generated")
                list_of_phenotypes_first_part.append(phenotype)
                list_of_arm_first_part.append(upscaled_arm_position)
                list_of_goal_first_part.append(upscaled_goal)
                list_of_third_coordinate.append(third_coordinate)

            # second part: only z coordinate
            for j in range(args.episodes):
                # rewrite coordinates and put them into big list
                upscaled_arm_position = []
                upscaled_goal = []
                for i in range(2):
                    # 1st and 2nd coordinate are desired goal coordinates,
                    # because result of first part must match start of second part
                    upscaled_arm_position.append(math.ceil(desired_goals[j][i] * 10))
                    upscaled_goal.append(math.ceil(desired_goals[j][i] * 10))
                # third coordinate stays the same for the first DT
                upscaled_arm_position.append(math.ceil(initial_goals[j][2] * 10))
                upscaled_goal.append(math.ceil(desired_goals[j][2] * 10))

                # print("Second part: arm: ")
                # print(upscaled_arm_position)
                # print("Second part: goal: ")
                # print(upscaled_goal)

                # generate current DT only once for every start-goal pair
                # DT is always working with 3 dimensions, even for FetchPush and FetchSlide
                # working here with 3D DT, dense is faster with 3D
                phenotype = dt.main(grid_size=20, agent_start=upscaled_arm_position, agent_goal=upscaled_goal,
                                    dimensions=3,
                                    reward_type="dense", obstacle_is_on=False)
                print("Phenotype part 2 number " + str(j) + " generated")
                list_of_phenotypes_second_part.append(phenotype)
                list_of_arm_second_part.append(upscaled_arm_position)
                list_of_goal_second_part.append(upscaled_goal)

        return list_of_phenotypes, list_of_arm, list_of_goal, list_of_third_coordinate, initial_goals, list_of_goal_first_part, list_of_arm_first_part, list_of_phenotypes_first_part, list_of_arm_second_part, list_of_goal_second_part, list_of_phenotypes_second_part

    def clip(self, value, min_value, max_value):
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    def learn(self, args, env, env_test, agent, buffer, list_of_phenotypes, list_of_arm, list_of_goal,
              list_of_third_coordinate, list_of_current_arm_position, list_of_goal_first_part, list_of_arm_first_part, list_of_phenotypes_first_part, list_of_arm_second_part, list_of_goal_second_part, list_of_phenotypes_second_part):
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
        goal_reached = False
        third_coordinate_is_done = False
        xy_is_done = False
        goal_reached_1 = []
        third_coordinate_is_done_1 = []
        xy_is_done_1 = []
        # every start-goal pair has its own "done" counter for obstacles
        for i in range(args.episodes):
            goal_reached_1.append(False)
            third_coordinate_is_done_1.append(False)
            xy_is_done_1.append(False)


        if args.env == "FetchPush-v1" or args.env == "FetchSlide-v1":
            # Q function values
            achieved_value = []
            if self.achieved_trajectory_pool.counter != 0:
                achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
                agent = self.args.agent
                for i in range(len(achieved_pool)):
                    obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
                           range(achieved_pool[i].shape[0])]
                    feed_dict = {
                        agent.raw_obs_ph: obs
                    }
                    value = agent.sess.run(agent.q_pi, feed_dict)[:, 0]
                    # value = np.clip(value, -1.0 / (1.0 - self.args.gamma), 0)
                    achieved_value.append(value.copy())
            if args.obstacle is True:
                # with obstacles we let every start-goal pair run separately
                # check if at least one goal has reached destination. goal_reached is set here
                for i in range(args.episodes):
                    # create a goal to compare intermediate goal to
                    current_goal = list_of_goal[i].copy()
                    current_goal = np.array(current_goal) / 10
                    current_goal = current_goal.tolist()
                    if args.env == "FetchPush-v1" or "FetchSlide-v1":
                        # if third coordinate was not appended yet
                        if len(current_goal) == 2:
                            current_goal.append(float(f'{list_of_third_coordinate[i]:.5f}'))
                        else:
                            current_goal[2] = float(f'{list_of_third_coordinate[i]:.5f}')

                    # preparation for equality check
                    tmp_arm = []
                    tmp_goal = []
                    for j in range(len(list_of_current_arm_position[i])):
                        # round, to prevent phantom decimal points like 0.7000000000000001
                        tmp_arm.append(round(list_of_current_arm_position[i][j], 10))
                        tmp_goal.append(current_goal[j])

                    # print("Temp arm: ")
                    # print(tmp_arm)
                    # print("Temp goal: ")
                    # print(tmp_goal)

                    # check point so the intermediate goal won't run away from the desired goal
                    if np.array_equal(tmp_goal, np.clip(tmp_arm, [0, 0, 0], tmp_goal)):
                        goal_reached_1[i] = True
                sum_q_vector_1 = 0
                # check if at least one mean_q is small enough. feedback_positive is set here
                for i in range(args.episodes):
                    # learner feedback basing on Q function values
                    if len(achieved_value) != 0:
                        q_vector = achieved_value[i]
                        sum_q_vector_2 = 0
                        for j in q_vector:
                            sum_q_vector_2 += j
                        # print("Mean Q single: " + str(sum_q_vector_2 / len(q_vector)))
                        sum_q_vector_1 += sum_q_vector_2

                # calculate mean over all vectors and episodes
                # sum / len(args.episodes) / len(q_vector)
                mean_q = sum_q_vector_1 / 50 / 51
                print("Mean Q over all episodes: " + str(mean_q))
                # If mean of all Q values is close enough to 0
                # -> learner feedback is positive. This synchronizes the steps
                result = abs(args.c - self.clip(mean_q, -1, 0))
                print("Result: " + str(result))

                for i in range(args.episodes):
                    obs = self.env_List[i].get_obs()
                    init_state = obs['observation'].copy()
                    explore_goal = self.sampler.sample(i)
                    intermediate_goal = []
                    # if goal is reached, then it would be automatically clipped.
                    # Clipping is now necessary because of dynamic step size
                    current_goal = list_of_goal[i].copy()
                    current_goal = np.array(current_goal) / 10
                    current_goal = current_goal.tolist()
                    if args.env == "FetchPush-v1" or "FetchSlide-v1":
                        # if third coordinate was not appended yet
                        if len(current_goal) == 2:
                            current_goal.append(float(f'{list_of_third_coordinate[i]:.5f}'))
                        else:
                            current_goal[2] = float(f'{list_of_third_coordinate[i]:.5f}')

                    # preparation for equality check
                    tmp_goal = []
                    for j in range(len(current_goal)):
                        tmp_goal.append(current_goal[j])

                    # check point so the intermediate goal won't run away from the desired goal
                    if goal_reached_1[i] is True:
                        # Clipping because of dynamic goals
                        # print("goal reached")
                        list_of_current_arm_position[i] = np.clip(list_of_current_arm_position[i].copy(), [0, 0, 0], tmp_goal)
                        intermediate_goal = np.array(list_of_current_arm_position[i].copy())
                    if goal_reached_1[i] is False:
                        # pass dt and current arm position to get next intermediate goal
                        # return 1 intermediate goal for every start-goal pair
                        intermediate_goal_1 = np.array(self.get_intermediate_goal(args, list_of_phenotypes[i],
                                                                                  list_of_current_arm_position[i].copy(),
                                                                                  list_of_third_coordinate[i].copy(),
                                                                                  1 - result))
                        intermediate_goal_2 = np.array(self.get_intermediate_goal(args, list_of_phenotypes[i],
                                                                                  intermediate_goal_1.copy(),
                                                                                  list_of_third_coordinate[i].copy(), result))
                        intermediate_goal_2_before_clip = intermediate_goal_2.copy()
                        # print("Before clip: ")
                        # print(intermediate_goal_2_before_clip)
                        intermediate_goal_2 = np.clip(intermediate_goal_2, [0, 0, 0], tmp_goal)
                        intermediate_goal_2_after_clip = intermediate_goal_2.copy()
                        # print("After clip: ")
                        # print(intermediate_goal_2_after_clip)
                        if np.array_equal(intermediate_goal_2_before_clip, intermediate_goal_2_after_clip) is False:
                            # this means that intermediate_goal_2 went too far and was clipped -> use only intermediate_goal_1
                            # CAREFUL: While testing with 1 DT this "if" can be triggered if wrong action is executed
                            # => clipping => inter_1 == inter_2 => average is getting left behind
                            intermediate_goal_2 = intermediate_goal_1.copy()

                        intermediate_goal = (intermediate_goal_1.copy() + intermediate_goal_2.copy()) / 2
                        # print("Inter 1: ")
                        # print(intermediate_goal_1)
                        # print("Inter 2: ")
                        # print(intermediate_goal_2)
                        # print("Average: ")
                        # print(intermediate_goal)
                        # Clipping because of dynamic goals
                        intermediate_goal = np.clip(intermediate_goal, [0, 0, 0], tmp_goal)

                    # write the intermediate goal
                    self.env_List[i].goal = np.array(intermediate_goal.copy())
                    # make the intermediate goal move
                    list_of_current_arm_position[i] = np.array(intermediate_goal.copy())

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
            else:
                # no obstacle
                # check if at least one goal has reached destination. goal_reached is set here
                for i in range(args.episodes):
                    # create a goal to compare intermediate goal to
                    current_goal = list_of_goal[i].copy()
                    current_goal = np.array(current_goal) / 10
                    current_goal = current_goal.tolist()
                    if args.env == "FetchPush-v1" or "FetchSlide-v1":
                        # if third coordinate was not appended yet
                        if len(current_goal) == 2:
                            current_goal.append(float(f'{list_of_third_coordinate[i]:.5f}'))
                        else:
                            current_goal[2] = float(f'{list_of_third_coordinate[i]:.5f}')

                    # preparation for equality check
                    tmp_arm = []
                    tmp_goal = []
                    for j in range(len(list_of_current_arm_position[i])):
                        # round, to prevent phantom decimal points like 0.7000000000000001
                        tmp_arm.append(round(list_of_current_arm_position[i][j], 10))
                        tmp_goal.append(current_goal[j])

                    # print("Temp arm: ")
                    # print(tmp_arm)
                    # print("Temp goal: ")
                    # print(tmp_goal)

                    # check point so the intermediate goal won't run away from the desired goal
                    if np.array_equal(tmp_goal, np.clip(tmp_arm, [0, 0, 0], tmp_goal)):
                        goal_reached = True
                sum_q_vector_1 = 0
                # check if at least one mean_q is small enough. feedback_positive is set here
                for i in range(args.episodes):
                    # learner feedback basing on Q function values
                    if len(achieved_value) != 0:
                        q_vector = achieved_value[i]
                        sum_q_vector_2 = 0
                        for j in q_vector:
                            sum_q_vector_2 += j
                        # print("Mean Q single: " + str(sum_q_vector_2 / len(q_vector)))
                        sum_q_vector_1 += sum_q_vector_2

                # calculate mean over all vectors and episodes
                # sum / len(args.episodes) / len(q_vector)
                mean_q = sum_q_vector_1 / 50 / 51
                print("Mean Q over all episodes: " + str(mean_q))
                # If mean of all Q values is close enough to 0
                # -> learner feedback is positive. This synchronizes the steps
                result = abs(args.c - self.clip(mean_q, -1, 0))
                print("Result: " + str(result))

                for i in range(args.episodes):
                    obs = self.env_List[i].get_obs()
                    init_state = obs['observation'].copy()
                    explore_goal = self.sampler.sample(i)
                    intermediate_goal = []
                    # if goal is reached, then it would be automatically clipped.
                    # Clipping is now necessary because of dynamic step size
                    current_goal = list_of_goal[i].copy()
                    current_goal = np.array(current_goal) / 10
                    current_goal = current_goal.tolist()
                    if args.env == "FetchPush-v1" or "FetchSlide-v1":
                        # if third coordinate was not appended yet
                        if len(current_goal) == 2:
                            current_goal.append(float(f'{list_of_third_coordinate[i]:.5f}'))
                        else:
                            current_goal[2] = float(f'{list_of_third_coordinate[i]:.5f}')

                    # preparation for equality check
                    tmp_goal = []
                    for j in range(len(current_goal)):
                        tmp_goal.append(current_goal[j])

                    # check point so the intermediate goal won't run away from the desired goal
                    if goal_reached is True:
                        # Clipping because of dynamic goals
                        # print("goal reached")
                        list_of_current_arm_position[i] = np.clip(list_of_current_arm_position[i].copy(), [0, 0, 0],
                                                                  tmp_goal)
                        intermediate_goal = np.array(list_of_current_arm_position[i].copy())
                    if goal_reached is False:
                        # pass dt and current arm position to get next intermediate goal
                        # return 1 intermediate goal for every start-goal pair
                        intermediate_goal_1 = np.array(self.get_intermediate_goal(args, list_of_phenotypes[i],
                                                                                  list_of_current_arm_position[
                                                                                      i].copy(),
                                                                                  list_of_third_coordinate[i].copy(),
                                                                                  1 - result))
                        intermediate_goal_2 = np.array(self.get_intermediate_goal(args, list_of_phenotypes[i],
                                                                                  intermediate_goal_1.copy(),
                                                                                  list_of_third_coordinate[i].copy(),
                                                                                  result))
                        intermediate_goal_2_before_clip = intermediate_goal_2.copy()
                        # print("Before clip: ")
                        # print(intermediate_goal_2_before_clip)
                        intermediate_goal_2 = np.clip(intermediate_goal_2, [0, 0, 0], tmp_goal)
                        intermediate_goal_2_after_clip = intermediate_goal_2.copy()
                        # print("After clip: ")
                        # print(intermediate_goal_2_after_clip)
                        if np.array_equal(intermediate_goal_2_before_clip, intermediate_goal_2_after_clip) is False:
                            # this means that intermediate_goal_2 went too far and was clipped -> use only intermediate_goal_1
                            # CAREFUL: While testing with 1 DT this "if" can be triggered if wrong action is executed
                            # => clipping => inter_1 == inter_2 => average is getting left behind
                            intermediate_goal_2 = intermediate_goal_1.copy()

                        intermediate_goal = (intermediate_goal_1.copy() + intermediate_goal_2.copy()) / 2
                        # print("Inter 1: ")
                        # print(intermediate_goal_1)
                        # print("Inter 2: ")
                        # print(intermediate_goal_2)
                        # print("Average: ")
                        # print(intermediate_goal)
                        # Clipping because of dynamic goals
                        intermediate_goal = np.clip(intermediate_goal, [0, 0, 0], tmp_goal)

                        # write the intermediate goal
                    self.env_List[i].goal = np.array(intermediate_goal.copy())
                    # make the intermediate goal move
                    list_of_current_arm_position[i] = np.array(intermediate_goal.copy())

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

        else:
            # Complex envs: FetchPickAndPlace and FetchReach
            # Q function values
            achieved_value = []
            if self.achieved_trajectory_pool.counter != 0:
                achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
                agent = self.args.agent
                for i in range(len(achieved_pool)):
                    obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
                           range(achieved_pool[i].shape[0])]
                    feed_dict = {
                        agent.raw_obs_ph: obs
                    }
                    value = agent.sess.run(agent.q_pi, feed_dict)[:, 0]
                    # value = np.clip(value, -1.0 / (1.0 - self.args.gamma), 0)
                    achieved_value.append(value.copy())

            # check if position is still in the range of DT N.1 or N.2
            for i in range(args.episodes):
                # create a goal to compare intermediate goal to
                current_goal_1 = list_of_goal_first_part[i].copy()
                current_goal_1 = np.array(current_goal_1) / 10
                current_goal_1 = current_goal_1.tolist()
                if len(current_goal_1) == 2:
                    current_goal_1.append(float(f'{list_of_third_coordinate[i]:.1f}'))
                # else:
                #     current_goal_1[2] = float(f'{list_of_third_coordinate[i]:.1f}')

                current_goal_2 = list_of_goal_second_part[i].copy()
                current_goal_2 = np.array(current_goal_2) / 10
                current_goal_2 = current_goal_2.tolist()

                # preparation for equality check
                tmp_arm = []
                tmp_goal_1 = []
                tmp_goal_2 = []
                for j in range(len(list_of_current_arm_position[i])):
                    # round, to prevent phantom decimal points like 0.7000000000000001
                    tmp_arm.append(round(list_of_current_arm_position[i][j], 10))
                    tmp_goal_1.append(current_goal_1[j])
                    tmp_goal_2.append(current_goal_2[j])

                # print("Temp arm: ")
                # print(tmp_arm)
                # print("Temp goal: ")
                # print(tmp_goal)

                # check point so the intermediate goal won't run away from the desired goal
                # Additionally this is needed to switch from calling first DT to second DT
                if np.array_equal(tmp_goal_2[:2], tmp_arm[:2]):
                    # checking for xy coordinates to match
                    # print("------------------------")
                    # print("xy is done")
                    # print("------------------------")
                    xy_is_done = True
                if tmp_goal_2[2] == tmp_arm[2]:
                    # checking for z coordinate to match
                    # print("------------------------")
                    # print("Third coordinate is done")
                    # print("------------------------")
                    third_coordinate_is_done = True
                if np.array_equal(tmp_goal_2, tmp_arm):
                    # checking for the end goal to be reached
                    # print("------------------------")
                    # print("goal reached")
                    # print("------------------------")
                    goal_reached = True
            sum_q_vector_1 = 0
            # check if at least one mean_q is small enough. feedback_positive is set here
            for i in range(args.episodes):
                # learner feedback basing on Q function values
                if len(achieved_value) != 0:
                    q_vector = achieved_value[i]
                    sum_q_vector_2 = 0
                    for j in q_vector:
                        sum_q_vector_2 += j
                    # print("Mean Q single: " + str(sum_q_vector_2 / len(q_vector)))
                    sum_q_vector_1 += sum_q_vector_2

            # calculate mean over all vectors and episodes
            # sum / len(args.episodes) / len(q_vector)
            mean_q = sum_q_vector_1 / 50 / 51
            print("Mean Q over all episodes: " + str(mean_q))
            # If mean of all Q values is close enough to 0
            # -> learner feedback is positive. This synchronizes the steps
            feedback = abs(args.c - self.clip(mean_q, -1, 0))
            print("Result: " + str(feedback))

            for i in range(args.episodes):
                obs = self.env_List[i].get_obs()
                init_state = obs['observation'].copy()
                explore_goal = self.sampler.sample(i)
                intermediate_goal = []
                # if goal is reached, then it would be automatically clipped.
                # Clipping is now necessary because of dynamic step size
                current_goal = list_of_goal_second_part[i].copy()
                current_goal = np.array(current_goal) / 10
                current_goal = current_goal.tolist()

                # preparation for equality check
                tmp_goal = []
                for j in range(len(current_goal)):
                    tmp_goal.append(current_goal[j])

                # if the end-goal isn't reached => call get_intermediate_goal
                intermediate_goal_1 = []
                intermediate_goal_2 = []
                if goal_reached is False:
                    # check for xy is done and z is done separately, because they can be done not simultaneously
                    if xy_is_done is False:
                        # pass dt and current arm position to get next intermediate goal
                        # return 1 intermediate goal for every start-goal pair
                        intermediate_goal_1_xy = np.array(self.get_intermediate_goal(args, list_of_phenotypes_first_part[i],
                                                                                  list_of_current_arm_position[i].copy(),
                                                                                  list_of_third_coordinate[i],
                                                                                  1 - feedback))
                        # print("intermediate_goal_1_xy: ")
                        # print(intermediate_goal_1_xy)
                    if third_coordinate_is_done is False:
                        intermediate_goal_1_z = np.array(self.get_intermediate_goal(args, list_of_phenotypes_second_part[i],
                                                                                  list_of_current_arm_position[i].copy(),
                                                                                  list_of_third_coordinate[i],
                                                                                  1 - feedback))
                        # print("intermediate_goal_1_z: ")
                        # print(intermediate_goal_1_z)
                    # combine xy and z goals to get the goal moving diagonally
                    # this if block is here to avoid referencing before assignment
                    if xy_is_done is False and third_coordinate_is_done is False:
                        intermediate_goal_1 = np.array([intermediate_goal_1_xy[0], intermediate_goal_1_xy[1], intermediate_goal_1_z[2]])
                    if xy_is_done is True and third_coordinate_is_done is False:
                        intermediate_goal_1 = np.array([list_of_current_arm_position[i][0], list_of_current_arm_position[i][1], intermediate_goal_1_z[2]])
                    # if third coordinate is done before xy; fixate third, so it doesn't run away
                    if xy_is_done is True and third_coordinate_is_done is True:
                        # this should not be triggered anyway, just for safety
                        intermediate_goal_1 = np.array([list_of_current_arm_position[i][0], list_of_current_arm_position[i][1], list_of_current_arm_position[i][2]])
                    if xy_is_done is False and third_coordinate_is_done is True:
                        intermediate_goal_1 = np.array([intermediate_goal_1_xy[0], intermediate_goal_1_xy[1], intermediate_goal_1_xy[2]])

                    # print("intermediate_goal_1: ")
                    # print(intermediate_goal_1)
                    if xy_is_done is False:
                        # same stuff for second intermediate goal for t+1.
                        # Is used later to create average from intermediate_goal_1 {t} and intermediate_goal_2 {t+1}
                        intermediate_goal_2_xy = np.array(self.get_intermediate_goal(args, list_of_phenotypes_first_part[i],
                                                                                  intermediate_goal_1.copy(),
                                                                                  list_of_third_coordinate[i],
                                                                                  feedback))
                        # print("intermediate_goal_2_xy: ")
                        # print(intermediate_goal_2_xy)
                    if third_coordinate_is_done is False:
                        intermediate_goal_2_z = np.array(self.get_intermediate_goal(args, list_of_phenotypes_second_part[i],
                                                                                  intermediate_goal_1.copy(),
                                                                                  list_of_third_coordinate[i],
                                                                                  feedback))
                        # print("intermediate_goal_2_z: ")
                        # print(intermediate_goal_2_z)
                        # combine xy and z goals to get the goal moving diagonally
                    if xy_is_done is False and third_coordinate_is_done is False:
                        intermediate_goal_2 = np.array(
                            [intermediate_goal_2_xy[0], intermediate_goal_2_xy[1], intermediate_goal_2_z[2]])
                    if xy_is_done is True and third_coordinate_is_done is False:
                        intermediate_goal_2 = np.array(
                            [list_of_current_arm_position[i][0], list_of_current_arm_position[i][1],
                             intermediate_goal_2_z[2]])
                    # if third coordinate is done before xy; fixate third, so it doesn't run away
                    if xy_is_done is True and third_coordinate_is_done is True:
                        # this should not be triggered anyway, just for safety
                        intermediate_goal_2 = np.array(
                            [list_of_current_arm_position[i][0], list_of_current_arm_position[i][1],
                             list_of_current_arm_position[i][2]])
                    if xy_is_done is False and third_coordinate_is_done is True:
                        intermediate_goal_2 = np.array(
                            [intermediate_goal_2_xy[0], intermediate_goal_2_xy[1], intermediate_goal_2_xy[2]])

                    # preparation for clip of intermediate_goal_2
                    # print("tmp-goal: ")
                    # print(tmp_goal)
                    # arr = [intermediate_goal_2.tolist(), tmp_goal]
                    # indexes_of_min_goal = np.argmin(arr, axis=0)
                    # indexes_of_max_goal = np.argmax(arr, axis=0)
                    #
                    # min_arr = np.array([arr[indexes_of_min_goal[0]][0], arr[indexes_of_min_goal[1]][1],
                    #                     arr[indexes_of_min_goal[2]][2]])
                    # max_arr = np.array([arr[indexes_of_max_goal[0]][0], arr[indexes_of_max_goal[1]][1],
                    #                     arr[indexes_of_max_goal[2]][2]])
                    # # this is here because of some python bugs. If variable is not used, it is getting distorted
                    # min_arr = min_arr
                    # max_arr = max_arr
                    # print("clipping of intermediate_goal_2:")
                    # print("min: ")
                    # print(min_arr)
                    # print("max: ")
                    # print(max_arr)
                    # max_arr = np.clip(max_arr.copy(), [0, 0, 0], tmp_goal)
                    # max_arr = np.clip(max_arr.copy(), [0, 0, 0], tmp_goal)
                    # intermediate_goal_2 = np.clip(intermediate_goal_2.copy(), min_arr, max_arr)
                    intermediate_goal_2 = np.clip(intermediate_goal_2.copy(), initial_goals[i], desired_goals[i])

                    intermediate_goal = (intermediate_goal_1.copy() + intermediate_goal_2.copy()) / 2

                    # Clipping because of dynamic goals
                    # preparation for clip of intermediate_goal
                    # arr = [intermediate_goal.tolist(), tmp_goal]
                    # indexes_of_min_goal = np.argmin(arr, axis=0)
                    # indexes_of_max_goal = np.argmax(arr, axis=0)
                    #
                    # min_arr = np.array([round(arr[indexes_of_min_goal[0]][0], 2), round(arr[indexes_of_min_goal[1]][1], 2),
                    #                     round(arr[indexes_of_min_goal[2]][2], 2)])
                    # max_arr = np.array([round(arr[indexes_of_max_goal[0]][0], 2), round(arr[indexes_of_max_goal[1]][1], 2),
                    #                     round(arr[indexes_of_max_goal[2]][2], 2)])
                    #
                    # min_arr = min_arr
                    # max_arr = max_arr
                    # print("clipping of intermediate_goal:")
                    # print("min: ")
                    # print(min_arr)
                    # print("max: ")
                    # print(max_arr)
                    intermediate_goal = np.clip(intermediate_goal.copy(), initial_goals[i], desired_goals[i])

                # this is the "end-goal reached" part
                if goal_reached is True:
                    # the goal does not move from here. It stays the same
                    list_of_current_arm_position[i] = np.clip(list_of_current_arm_position[i].copy(), [0, 0, 0], tmp_goal)
                    intermediate_goal = np.array(list_of_current_arm_position[i].copy())

                # write the intermediate goal
                self.env_List[i].goal = np.array(intermediate_goal.copy())
                # make the intermediate goal move
                list_of_current_arm_position[i] = np.array(intermediate_goal.copy())

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
        print("Intermediate goals: ")
        print(list_of_current_arm_position)

        selection_trajectory_idx = {}
        for i in range(self.args.episodes):
            if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())

        return list_of_current_arm_position
