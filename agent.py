from os import path
from env import DiscretizedAgentPose
import random
from env import (
    ManyObjectsEnv,
    ActionsInv,
)
import matplotlib.pyplot as plt
from utils import astar, get_weighted_adjacency_list
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import Any


def normalize_target(y: dict[Any, float]) -> dict[Any, float]:
    y_array = np.array(list(y.values()))
    y_min = y_array.min()
    y_max = y_array.max()
    if y_max - y_min == 0:
        return {key: 0.5 for key in y}
    normalized_y = {key: (value - y_min) / (y_max - y_min) for key, value in y.items()}
    return normalized_y


class RFModel:
    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=0,
        )

    def fit(self, X: list[DiscretizedAgentPose], y: list[float]):
        X_int = [(pose.idx_x, pose.idx_z, pose.idx_yaw, pose.idx_h) for pose in X]
        self.model.fit(X_int, y)

    def predict(self, X: list[DiscretizedAgentPose]) -> list[float]:
        X_int = [(pose.idx_x, pose.idx_z, pose.idx_yaw, pose.idx_h) for pose in X]
        predictions = self.model.predict(X_int).tolist()
        return predictions

    def ucb_predict(self, X: list[DiscretizedAgentPose]) -> list[float]:
        X_int = [(pose.idx_x, pose.idx_z, pose.idx_yaw, pose.idx_h) for pose in X]

        all_tree_predictions = np.array(
            [tree.predict(X_int) for tree in self.model.estimators_]
        )

        y_mean = all_tree_predictions.mean(axis=0)
        y_std = all_tree_predictions.std(axis=0)
        ucb_values = y_mean + 2 * y_std
        return ucb_values.tolist()


class Optimizer:
    def __init__(
        self,
        surrogate_model,
        input_space: list[DiscretizedAgentPose],
    ):
        self.surrogate_model = surrogate_model
        self.input_space = input_space
        self.xi = []
        self.yi = []

        self.pose2int: dict[DiscretizedAgentPose, int] = {}
        self.int2pose: dict[int, DiscretizedAgentPose] = {}

        for i, pose_tuple in enumerate(self.input_space):
            self.pose2int[pose_tuple] = i
            self.int2pose[i] = pose_tuple

    def update(self, x: list[DiscretizedAgentPose], y: list[float]):
        self.xi.extend(x)
        self.yi.extend(y)

    def sample_input_space(
        self, visited: set[DiscretizedAgentPose], num_samples: int
    ) -> list[DiscretizedAgentPose]:
        unvisited_poses = [pose for pose in self.input_space if pose not in visited]
        sampled_poses = random.sample(unvisited_poses, k=num_samples)
        return sampled_poses

    def get_values(self) -> dict[DiscretizedAgentPose, float]:
        self.surrogate_model.fit(self.xi, self.yi)
        y_pred = self.surrogate_model.predict(self.input_space)
        values = {pose: y for pose, y in zip(self.input_space, y_pred)}
        return values

    def get_ucb_values(self) -> dict[DiscretizedAgentPose, float]:
        self.surrogate_model.fit(self.xi, self.yi)
        y_pred = self.surrogate_model.ucb_predict(self.input_space)
        values = {pose: y for pose, y in zip(self.input_space, y_pred)}
        return values

    def acquisition_function(
        self,
        x: DiscretizedAgentPose,
        current_position: DiscretizedAgentPose,
        ucb_w: dict[int, list[tuple[int, float, int]]],
        ucb_v: dict[int, float],
    ) -> float:

        if x == current_position:
            return 0.0

        int_curr = self.pose2int[current_position]
        int_x = self.pose2int[x]

        int_path = astar(
            start=int_curr,
            goal=int_x,
            w=ucb_w,
            int2pose=self.int2pose,
        )

        if len(int_path) == 0:
            raise ValueError("Path is empty")

        path_value = sum(ucb_v[node] for action, node in int_path)

        return path_value / len(int_path)

    def suggest(
        self,
        current_position: DiscretizedAgentPose,
        visited_positions: set[DiscretizedAgentPose],
        adjacency_list: dict[
            DiscretizedAgentPose, list[tuple[int, DiscretizedAgentPose]]
        ],
        num_samples: int = 10,
    ) -> list[tuple[int, DiscretizedAgentPose]]:
        assert len(self.xi) > 0, "No data to build surrogate model"

        # fit and evaluate on the whole input space
        ucb_values = self.get_ucb_values()
        normalized_ucb_values = normalize_target(ucb_values)

        # build the weighted adjacency list
        ucb_weighted_adjacency_list = get_weighted_adjacency_list(
            normalized_ucb_values, adjacency_list, visited_positions
        )
        ucb_w = {
            self.pose2int[pose]: [
                (action, weight, self.pose2int[neighbor])
                for action, weight, neighbor in neighbors
            ]
            for pose, neighbors in ucb_weighted_adjacency_list.items()
        }
        ucb_v = {
            self.pose2int[pose]: value for pose, value in normalized_ucb_values.items()
        }

        # sample next point
        X = self.sample_input_space(visited_positions, num_samples=num_samples)
        acquisitions = {
            x: self.acquisition_function(
                x,
                current_position,
                ucb_w,
                ucb_v,
            )
            for x in X
        }
        x = max(acquisitions, key=acquisitions.get)  # type: ignore

        # find shortest path to x
        int_curr = self.pose2int[current_position]
        int_x = self.pose2int[x]
        int_path = astar(
            start=int_curr,
            goal=int_x,
            w=ucb_w,
            int2pose=self.int2pose,
        )
        return [(action, self.int2pose[i]) for action, i in int_path]


class BOAgent:
    def __init__(
        self,
        surrogate_model,
        input_space: list[DiscretizedAgentPose],
        env: ManyObjectsEnv,
        num_samples: int = 100,
        init_info: dict = None,
    ):

        self.optimizer = Optimizer(
            surrogate_model=surrogate_model,
            input_space=input_space,
        )
        self.input_space = input_space
        self.env = env
        self.num_samples = num_samples

        self.visited_positions: set[DiscretizedAgentPose] = set()
        self.actions_registry: list[str] = []

        self.update(env.get_pose(), init_info["state_value"])

    def act(self) -> str:
        if not self.actions_registry:
            path = self.optimizer.suggest(
                current_position=self.env.get_pose(),
                visited_positions=self.visited_positions,
                adjacency_list=self.env.adjacency_list,
                num_samples=self.num_samples,
            )
            visited_fraction = sum(
                [p in self.visited_positions for p, _ in path]
            ) / len(path)
            print(
                f"Visited fraction in path: {visited_fraction:.2f}, size of visited set: {len(self.visited_positions)}"
            )
            self.actions_registry = [ActionsInv[a] for a, _ in path]
            self.evaluate(self.env.ground_truth_values)

        action_str = self.actions_registry.pop(0)

        return action_str

    def update(self, current_pose: DiscretizedAgentPose, state_value: float):
        self.visited_positions.add(current_pose)
        self.optimizer.update([current_pose], [state_value])

    def evaluate(self, ground_truth_values: dict[DiscretizedAgentPose, float]) -> float:
        values = self.optimizer.get_values()

        mse_train = np.mean(
            [
                (values[pose] - ground_truth_values[pose]) ** 2
                for pose in self.visited_positions
            ]
        )
        print(f"RF Model Train MSE: {mse_train:.4f}")

        mse = np.mean(
            [
                (values[pose] - ground_truth_values[pose]) ** 2
                for pose in self.input_space
            ]
        )
        print(f"RF Model MSE: {mse:.4f}")

        visited_array = self.get_visited_array()
        plt.imshow(visited_array, cmap="gray")
        plt.title("Visited Positions")
        plt.savefig(path.join("results", "visited_positions.png"))
        plt.close()

        return float(mse)

    def get_visited_array(self) -> np.ndarray:
        visited_array = np.zeros((16, 16), dtype=bool)
        for pose in self.visited_positions:
            idx_x, idx_z = pose.idx_x, pose.idx_z
            visited_array[idx_x, idx_z] = True

        return visited_array
