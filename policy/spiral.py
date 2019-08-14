import numpy as np
import gym_cap.envs.const as const
import math
from policy.policy import Policy

class Spiral(Policy):

    def __init__(self):
        super().__init__()

    def initiate(self, free_map, agent_list):
        super().initiate(free_map, agent_list)

        self.enemy_range = 5  # Range to see around and avoid enemy
        self.flag_range = 5  # Range to see the flag

        self.found_route = []
        self.agent_route = []

        self.agent_steps = [0] * len(agent_list)

        for idx, agent in enumerate(agent_list):
            start = agent.get_loc()
            self.agent_route.append(self.spiral(start))
            self.found_route.append(self.agent_route[idx] is not None)

    def gen_action(self, agent_list, observation, free_map=None):
        action_out = []
        for idx, agent in enumerate(agent_list):
            if not agent.isAlive or not self.found_route[idx]:
                action_out.append(0)
                continue

            cur_loc = agent.get_loc()
            if self.agent_route[idx][self.agent_steps[idx]] != cur_loc:
                self.agent_steps[idx] += 1
            cur_step = self.agent_steps[idx]

            if cur_step >= len(self.agent_route[idx]):
                action_out.append(0)
                continue
            new_loc = self.agent_route[idx][cur_step + 1]
            action = self.move_toward(cur_loc, new_loc)
            action_out.append(action)

        return action_out

    def spiral(self, loc):
        route = [loc]
        cur_idx = 0
        mapx, mapy = self.free_map.shape

        visit = np.full(shape=(mapx, mapy), fill_value=0)  # 0 is unknown, 1 is visited
        visit[self.free_map == 8] = 1

        def blocking(position,d):
            nx, ny = self.next_loc(position, d)
            if nx < 0 or nx >= mapx: return True
            elif ny < 0 or ny >= mapy: return True

            return visit[nx][ny] == 1

        while True:
            initial = x, y = route[cur_idx]
            visit[x][y] = 1

            action_pool = [move for move in range(1, 5) if not blocking(initial, move)]
            if action_pool == []:   # no possible moves for agent
                route.append(route[cur_idx])
                return route

            # out of possible moves, minimize distance from origin
            minDist = None
            final_location = None

            for move in action_pool[::-1]:
                new_loc = self.next_loc(initial, move)
                dist = self.distance(loc, new_loc, True)

                if minDist is None:
                    minDist = dist
                    final_location = new_loc
                elif dist < minDist:
                    minDist = dist
                    final_location = new_loc

            route.append(final_location)
            cur_idx += 1
