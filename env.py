from miniworld.entity import MeshEnt
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS
import math
from dataclasses import dataclass
import random

Actions = {
    "MoveAhead": 0,
    "MoveBack": 1,
    "MoveLeft": 2,
    "MoveRight": 3,
    "RotateLeft": 4,
    "RotateRight": 5,
    "LookUp": 6,
    "LookDown": 7,
}
ActionsInv = {v: k for k, v in Actions.items()}
GRID_STEP = 1
V_ANGLE_STEP = 45
H_ANGLE_STEP = 30
V_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
H_ANGLES = [-30, 0, 30]
V_ANGLES_INV = {angle: idx for idx, angle in enumerate(V_ANGLES)}
H_ANGLES_INV = {angle: idx for idx, angle in enumerate(H_ANGLES)}

DELTA = {
    0: (GRID_STEP, 0),
    90: (0, GRID_STEP),
    180: (-GRID_STEP, 0),
    270: (0, -GRID_STEP),
}

Discretized_AgentPose = tuple[int, int, int, int]


@dataclass
class AgentPose:
    x: float
    y: float
    z: float
    yaw: float
    camera_horizon: float

    def __init__(self, x, y, z, yaw, camera_horizon):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.camera_horizon = camera_horizon

    def discretize(self) -> Discretized_AgentPose:
        idx_x = int(self.x // GRID_STEP)
        idx_z = int(self.z // GRID_STEP)
        idx_yaw = V_ANGLES_INV[int(self.yaw % 360)]
        idx_h = H_ANGLES_INV[int(self.camera_horizon)]
        return (idx_x, idx_z, idx_yaw, idx_h)

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.yaw, self.camera_horizon))

    def get_neighbors(
        self, max_x: float, max_z: float
    ) -> list[tuple[str, "AgentPose"]]:
        neighbors = []

        # if agent is pointing diagonally, adjust movement accordingly
        move_angle = int(self.yaw % 360)
        if move_angle % 90 == 45:
            move_angle = int(self.yaw % 360) - 45

        # movements
        for (dx, dz), action_str in [
            (DELTA[move_angle], "MoveAhead"),
            (DELTA[(move_angle + 180) % 360], "MoveBack"),
            (DELTA[(move_angle - 90) % 360], "MoveLeft"),
            (DELTA[(move_angle + 90) % 360], "MoveRight"),
        ]:
            if 0 <= self.x + dx < max_x and 0 <= self.z + dz < max_z:
                neighbors.append(
                    (
                        action_str,
                        AgentPose(
                            self.x + dx,
                            self.y,
                            self.z + dz,
                            self.yaw,
                            self.camera_horizon,
                        ),
                    )
                )

        # rotation left / right
        neighbors.append(
            (
                "RotateLeft",
                AgentPose(
                    self.x,
                    self.y,
                    self.z,
                    (self.yaw + V_ANGLE_STEP) % 360,
                    self.camera_horizon,
                ),
            )
        )
        neighbors.append(
            (
                "RotateRight",
                AgentPose(
                    self.x,
                    self.y,
                    self.z,
                    (self.yaw - V_ANGLE_STEP) % 360,
                    self.camera_horizon,
                ),
            )
        )

        # look up / down
        if self.camera_horizon > H_ANGLES[0]:
            neighbors.append(
                (
                    "LookDown",
                    AgentPose(
                        self.x,
                        self.y,
                        self.z,
                        self.yaw,
                        self.camera_horizon - H_ANGLE_STEP,
                    ),
                )
            )
        if self.camera_horizon < H_ANGLES[-1]:
            neighbors.append(
                (
                    "LookUp",
                    AgentPose(
                        self.x,
                        self.y,
                        self.z,
                        self.yaw,
                        self.camera_horizon + H_ANGLE_STEP,
                    ),
                )
            )

        return neighbors


def agent_pose_from_discretized(
    idx_x: int, idx_z: int, idx_yaw: int, idx_h: int
) -> AgentPose:
    return AgentPose(
        x=GRID_STEP * idx_x,
        y=0,
        z=GRID_STEP * idx_z,
        yaw=V_ANGLES[idx_yaw],
        camera_horizon=H_ANGLES[idx_h],
    )


def dijkstra(
    start: int,
    goal: int,
    w: dict[int, list[tuple[int, float, int]]],
) -> list[tuple[int, int]]:
    """
    w is the weighted_adjacency_list: dict[node, list[(action, cost, neighbor)]]
    Return the list of (action, node) representing the path.
    """
    import heapq

    pq = [(0.0, start)]

    distances = {start: 0.0}

    parent_map = {start: (None, None)}

    while pq:
        current_cost, u = heapq.heappop(pq)

        # If we reached the goal, reconstruct the path
        if u == goal:
            path = []
            curr = goal
            while curr is not None:
                parent, action = parent_map[curr]
                if action is not None:
                    path.append((action, curr))
                curr = parent
            return path[::-1]  # Reverse to get path from start to goal

        # If we found a better way to u already, skip this entry
        if current_cost > distances.get(u, float("inf")):
            continue

        # Explore neighbors
        for action, edge_cost, v in w.get(u, []):
            new_cost = current_cost + edge_cost

            if v not in distances or new_cost < distances[v]:
                distances[v] = new_cost
                parent_map[v] = (u, action)  # type: ignore
                heapq.heappush(pq, (new_cost, v))

    return []  # Return empty list if no path exists


def get_weighted_adjacency_list(
    values: dict[Discretized_AgentPose, float],
    adjacency_list: dict[
        Discretized_AgentPose, list[tuple[int, Discretized_AgentPose]]
    ],
    visited_positions: set[Discretized_AgentPose],
) -> dict[Discretized_AgentPose, list[tuple[int, float, Discretized_AgentPose]]]:
    weighted_adjacency_list = {}

    for node, neighbors in adjacency_list.items():
        weighted_adjacency_list[node] = []

        for action, n_node_tuple in neighbors:
            value_n = values[n_node_tuple]
            assert 0.0 <= value_n <= 1.0
            cost = 1 - value_n if node in visited_positions else 2.0 + value_n
            weighted_adjacency_list[node].append((action, cost, n_node_tuple))

    return weighted_adjacency_list


class ManyObjectsEnv(MiniWorldEnv):
    def __init__(self, n, grid_size, **kwargs) -> None:
        self.n = n
        assert grid_size >= 2
        self.grid_size = grid_size

        super().__init__(max_episode_steps=100_000, **kwargs)

    def _gen_world(self) -> None:

        self.add_rect_room(
            min_x=0,
            max_x=self.grid_size,
            min_z=0,
            max_z=self.grid_size,
            wall_tex="cinder_blocks",
            floor_tex="slime",
        )

        for _ in range(self.n):
            self.box = self.place_entity(
                MeshEnt(mesh_name="medkit", height=0.50, static=True)
            )

        # Place the agent
        self.place_agent(
            pos=(0, 0, 0),
            dir=0,
        )
        self.teleport_agent(random.choice(self.enumerate_poses()))
        self.step_count = 0

        # initialize graph
        self.adjacency_list: dict[
            Discretized_AgentPose, list[tuple[int, Discretized_AgentPose]]
        ] = {}

        for pose in self.enumerate_poses():
            pose_tuple = pose.discretize()
            self.adjacency_list[pose_tuple] = []
            pose = agent_pose_from_discretized(*pose_tuple)
            neighbors = pose.get_neighbors(self.grid_size, self.grid_size)

            for action_str, n_pose in neighbors:
                n_pose_tuple = n_pose.discretize()
                self.adjacency_list[pose_tuple].append(
                    (Actions[action_str], n_pose_tuple)
                )

    def get_pose(self) -> AgentPose:
        x, y, z = self.agent.pos
        yaw = (self.agent.dir * 180 / math.pi) % 360
        camera_horizon = self.agent.cam_pitch
        return AgentPose(x, y, z, yaw, camera_horizon)

    def move_agent(self, dx, dy) -> bool:
        self.agent.pos = (
            self.agent.pos[0] + dx,
            self.agent.pos[1],
            self.agent.pos[2] + dy,
        )
        return True

    def turn_agent(self, turn_angle) -> bool:
        turn_angle *= math.pi / 180
        self.agent.dir += turn_angle
        self.agent.dir %= 2 * math.pi
        return True

    def pitch_agent(self, pitch_angle) -> bool:
        if (
            self.agent.cam_pitch + pitch_angle > 30
            or self.agent.cam_pitch + pitch_angle < -30
        ):
            return False

        self.agent.cam_pitch += pitch_angle
        return True

    def enumerate_poses(self) -> list[AgentPose]:
        poses = []

        for idx_x in range(int(self.grid_size / GRID_STEP)):
            for idx_z in range(int(self.grid_size / GRID_STEP)):
                for idx_yaw in range(len(V_ANGLES)):
                    for idx_h in range(len(H_ANGLES)):
                        pose = AgentPose(
                            x=GRID_STEP * idx_x,
                            y=0,
                            z=GRID_STEP * idx_z,
                            yaw=V_ANGLES[idx_yaw],
                            camera_horizon=H_ANGLES[idx_h],
                        )
                        poses.append(pose)
        return poses

    def teleport_agent(self, pose: AgentPose):
        self.agent.pos = (pose.x, pose.y, pose.z)
        self.agent.dir = pose.yaw * math.pi / 180
        self.agent.cam_pitch = pose.camera_horizon

    def step(self, action_str: str) -> tuple:
        self.step_count += 1

        yaw = self.get_pose().yaw
        move_angle = int(yaw % 360)
        if move_angle % 90 == 45:
            move_angle = int(yaw % 360) - 45

        if action_str == "MoveAhead":
            dx, dy = DELTA[move_angle]
            self.move_agent(dx, dy)

        elif action_str == "MoveBack":
            dx, dy = DELTA[int((move_angle + 180) % 360)]
            self.move_agent(dx, dy)

        elif action_str == "MoveLeft":
            dx, dy = DELTA[int((move_angle - 90) % 360)]
            self.move_agent(dx, dy)

        elif action_str == "MoveRight":
            dx, dy = DELTA[int((move_angle + 90) % 360)]
            self.move_agent(dx, dy)

        elif action_str == "RotateLeft":
            self.turn_agent(V_ANGLE_STEP)

        elif action_str == "RotateRight":
            self.turn_agent(-V_ANGLE_STEP)

        elif action_str == "LookUp":
            self.pitch_agent(H_ANGLE_STEP)

        elif action_str == "LookDown":
            self.pitch_agent(-H_ANGLE_STEP)

        else:
            raise ValueError(f"Unknown action: {action_str}")
        obs = None

        reward = 0
        termination = False
        truncation = False
        info = {}

        info["state_value"] = len(self.get_visible_ents())

        return obs, reward, termination, truncation, info

    def reset(self) -> tuple:
        super().reset()

        info = {}
        info["state_value"] = len(self.get_visible_ents())
        return (None, info)
