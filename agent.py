from env import Discretized_AgentPose
import random
from env import (
    ManyObjectsEnv,
    dijkstra,
    get_weighted_adjacency_list,
    ActionsInv,
    agent_pose_from_discretized,
)
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RFModel:
    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=0,
        )

    def normalize_target(self, y: list[float]) -> list[float]:
        y_array = np.array(y)
        y_min = y_array.min()
        y_max = y_array.max()
        if y_max - y_min == 0:
            return [0.5 for _ in y]
        normalized = (y_array - y_min) / (y_max - y_min)
        return normalized.tolist()

    def fit(self, X: list[list[float]], y: list[float]):
        normalized_targets = self.normalize_target(y)
        self.model.fit(X, normalized_targets)

    def predict(self, X: list[list[float]]) -> list[float]:
        predictions = self.model.predict(X).tolist()
        return predictions

    def ucb_predict(self, X: list[list[float]]) -> list[float]:
        all_tree_predictions = np.array(
            [tree.predict(X) for tree in self.model.estimators_]
        )

        y_mean = all_tree_predictions.mean(axis=0)
        y_std = all_tree_predictions.std(axis=0)
        ucb_values = y_mean + y_std
        ucb_values = np.clip(ucb_values, 0.0, 1.0)
        return ucb_values.tolist()


class Optimizer:
    def __init__(
        self,
        surrogate_model,
        input_space: list[Discretized_AgentPose],
    ):
        self.surrogate_model = surrogate_model
        self.input_space = input_space
        self.xi = []
        self.yi = []

        self.pose2int: dict[Discretized_AgentPose, int] = {}
        self.int2pose: dict[int, Discretized_AgentPose] = {}

        for i, pose_tuple in enumerate(self.input_space):
            self.pose2int[pose_tuple] = i
            self.int2pose[i] = pose_tuple

    def update(self, x: list[Discretized_AgentPose], y: list[float]):
        self.xi.extend(x)
        self.yi.extend(y)

    def shortest_path(
        self,
        start: Discretized_AgentPose,
        goal: Discretized_AgentPose,
        weighted_adjacency_list: dict[
            Discretized_AgentPose, list[tuple[int, float, Discretized_AgentPose]]
        ],
        values: dict[Discretized_AgentPose, float],
    ) -> tuple[list[tuple[int, Discretized_AgentPose]], float]:
        int_start = self.pose2int[start]
        int_goal = self.pose2int[goal]

        if int_start == int_goal:
            return [(-1, start)], 0.0

        w = {
            self.pose2int[k]: [(a, cost, self.pose2int[n]) for a, cost, n in v]
            for k, v in weighted_adjacency_list.items()
        }
        int_path = dijkstra(int_start, int_goal, w)
        path = list(map(lambda x: (x[0], self.int2pose[x[1]]), int_path))

        # compute path value and travel cost
        path_value = 0

        for a, pose in path:
            path_value += values[pose]

        return path, path_value

    def sample_input_space(
        self, visited: set[Discretized_AgentPose], num_samples: int
    ) -> list[Discretized_AgentPose]:
        unvisited_poses = [pose for pose in self.input_space if pose not in visited]
        sampled_poses = random.sample(unvisited_poses, k=num_samples)
        return sampled_poses

    def get_values(self) -> dict[Discretized_AgentPose, float]:
        self.surrogate_model.fit(self.xi, self.yi)
        X = [list(x) for x in self.input_space]
        y_pred = self.surrogate_model.predict(X)
        values = {pose: y for pose, y in zip(self.input_space, y_pred)}
        return values

    def acquisition_function(
        self,
        x: Discretized_AgentPose,
        current_position: Discretized_AgentPose,
        ucb_weighted_adjacency_list: dict[
            Discretized_AgentPose, list[tuple[int, float, Discretized_AgentPose]]
        ],
        ucb_values: dict[Discretized_AgentPose, float],
    ) -> float:

        if x == current_position:
            return float("-inf")

        path, path_value = self.shortest_path(
            start=current_position,
            goal=x,
            weighted_adjacency_list=ucb_weighted_adjacency_list,
            values=ucb_values,
        )

        if len(path) == 0:
            raise ValueError("Path is empty")

        return path_value / len(path)

    def suggest(
        self,
        current_position: Discretized_AgentPose,
        visited_positions: set[Discretized_AgentPose],
        adjacency_list: dict[
            Discretized_AgentPose, list[tuple[int, Discretized_AgentPose]]
        ],
    ) -> list[tuple[int, Discretized_AgentPose]]:
        assert len(self.xi) > 0, "No data to build surrogate model"

        # fit and evaluate on the whole input space
        normalized_ucb_values = self.get_values()

        # build the weighted adjacency list
        weighted_adjacency_list = get_weighted_adjacency_list(
            normalized_ucb_values, adjacency_list, visited_positions
        )

        # sample next point
        X = self.sample_input_space(visited_positions, num_samples=10)
        acquisitions = {
            x: self.acquisition_function(
                x,
                current_position,
                weighted_adjacency_list,
                normalized_ucb_values,
            )
            for x in X
        }
        print((current_position, acquisitions))
        x = max(acquisitions, key=acquisitions.get)  # type: ignore

        # find shortest path to x
        path, path_value = self.shortest_path(
            start=current_position,
            goal=x,
            weighted_adjacency_list=weighted_adjacency_list,
            values=normalized_ucb_values,
        )
        return path


class BOAgent:
    def __init__(
        self,
        surrogate_model,
        input_space: list[Discretized_AgentPose],
        env: ManyObjectsEnv,
        seed: int = 0,
    ):
        random.seed(seed)

        self.optimizer = Optimizer(
            surrogate_model=surrogate_model,
            input_space=input_space,
        )
        self.input_space = input_space
        self.env = env

        self.visited_positions: set[Discretized_AgentPose] = set()
        self.actions_registry: list[str] = []

        self.visited_positions.add(self.env.get_pose().discretize())

    def act(self) -> str:
        if not self.actions_registry:
            path = self.optimizer.suggest(
                current_position=self.env.get_pose().discretize(),
                visited_positions=self.visited_positions,
                adjacency_list=self.env.adjacency_list,
            )
            self.actions_registry = [ActionsInv[a] for a, _ in path]

        action_str = self.actions_registry.pop(0)

        return action_str

    def update(self, state_value: float):
        current_pose = self.env.get_pose().discretize()
        self.visited_positions.add(current_pose)
        self.optimizer.update([current_pose], [state_value])
