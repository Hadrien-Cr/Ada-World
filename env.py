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
    0: (1, 0),
    90: (0, 1),
    180: (-1, 0),
    270: (0, -1),
}


@dataclass
class DiscretizedAgentPose:
    idx_x: int
    idx_z: int
    idx_yaw: int
    idx_h: int

    def __init__(self, idx_x, idx_z, idx_yaw, idx_h) -> None:
        self.idx_x = idx_x
        self.idx_z = idx_z
        self.idx_yaw = idx_yaw
        self.idx_h = idx_h

    def __hash__(self):
        return hash((self.idx_x, self.idx_z, self.idx_yaw, self.idx_h))

    def get_neighbors(
        self, max_idx_x: int, max_idx_z: int
    ) -> list[tuple[str, "DiscretizedAgentPose"]]:
        neighbors = []

        # if agent is pointing diagonally, adjust movement accordingly
        if self.idx_yaw % 2 == 0:
            move_angle = V_ANGLES[self.idx_yaw] % 360
        else:
            move_angle = (V_ANGLES[self.idx_yaw] - 45) % 360

        # movements
        for (dx, dz), action_str in [
            (DELTA[move_angle], "MoveAhead"),
            (DELTA[(move_angle + 180) % 360], "MoveBack"),
            (DELTA[(move_angle - 90) % 360], "MoveLeft"),
            (DELTA[(move_angle + 90) % 360], "MoveRight"),
        ]:
            if 0 <= self.idx_x + dx < max_idx_x and 0 <= self.idx_z + dz < max_idx_z:
                neighbors.append(
                    (
                        action_str,
                        DiscretizedAgentPose(
                            self.idx_x + dx,
                            self.idx_z + dz,
                            self.idx_yaw,
                            self.idx_h,
                        ),
                    )
                )

        # rotation left / right
        neighbors.append(
            (
                "RotateLeft",
                DiscretizedAgentPose(
                    self.idx_x,
                    self.idx_z,
                    (self.idx_yaw + 1) % len(V_ANGLES),
                    self.idx_h,
                ),
            )
        )
        neighbors.append(
            (
                "RotateRight",
                DiscretizedAgentPose(
                    self.idx_x,
                    self.idx_z,
                    (self.idx_yaw - 1) % len(V_ANGLES),
                    self.idx_h,
                ),
            )
        )

        # look up / down
        if self.idx_h > 0:
            neighbors.append(
                (
                    "LookDown",
                    DiscretizedAgentPose(
                        self.idx_x,
                        self.idx_z,
                        self.idx_yaw,
                        self.idx_h - 1,
                    ),
                )
            )
        if self.idx_h < len(H_ANGLES) - 1:
            neighbors.append(
                (
                    "LookUp",
                    DiscretizedAgentPose(
                        self.idx_x,
                        self.idx_z,
                        self.idx_yaw,
                        self.idx_h + 1,
                    ),
                )
            )

        return neighbors


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

        self.place_agent(
            pos=(0, 0, 0),
            dir=0,
        )

        # initialize graph
        self.adjacency_list: dict[
            DiscretizedAgentPose, list[tuple[int, DiscretizedAgentPose]]
        ] = {}

        for pose in self.enumerate_poses():
            self.adjacency_list[pose] = []
            neighbors = pose.get_neighbors(self.grid_size, self.grid_size)

            for action_str, n_pose in neighbors:
                self.adjacency_list[pose].append((Actions[action_str], n_pose))

    def initialize_gt_values(self) -> None:
        self.ground_truth_values: dict[DiscretizedAgentPose, float] = {}

        for pose in self.enumerate_poses():
            self.teleport_agent(pose)
            state_value = self.get_state_value()
            self.ground_truth_values[pose] = state_value

    def get_pose(self) -> DiscretizedAgentPose:
        x, y, z = self.agent.pos  # type: ignore
        yaw = (self.agent.dir * 180 / math.pi) % 360  # type: ignore
        camera_horizon = self.agent.cam_pitch
        idx_x = round(x / GRID_STEP)
        idx_z = round(z / GRID_STEP)
        idx_yaw = V_ANGLES_INV[round(yaw)]
        idx_h = H_ANGLES_INV[round(camera_horizon)]

        return DiscretizedAgentPose(idx_x, idx_z, idx_yaw, idx_h)

    def move_agent(self, dx, dy) -> bool:
        x, y, z = self.agent.pos  # type: ignore
        self.agent.pos = (x + dx, y, z + dy)  # type: ignore
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

    def enumerate_poses(self) -> list[DiscretizedAgentPose]:
        return [
            DiscretizedAgentPose(idx_x, idx_z, idx_yaw, idx_h)
            for idx_x in range(self.grid_size)
            for idx_z in range(self.grid_size)
            for idx_yaw in range(len(V_ANGLES))
            for idx_h in range(len(H_ANGLES))
        ]

    def teleport_agent(self, pose: DiscretizedAgentPose):
        self.agent.pos = (pose.idx_x * GRID_STEP, 0, pose.idx_z * GRID_STEP)  # type: ignore
        self.agent.dir = V_ANGLES[pose.idx_yaw] * math.pi / 180  # type: ignore
        self.agent.cam_pitch = H_ANGLES[pose.idx_h] * math.pi / 180  # type: ignore

    def step(self, action_str: str) -> tuple:
        self.step_count += 1
        pose = self.get_pose()

        if pose.idx_yaw % 2 == 0:
            move_angle = V_ANGLES[pose.idx_yaw] % 360
        else:
            move_angle = (V_ANGLES[pose.idx_yaw] - 45) % 360

        if action_str == "MoveAhead":
            dx, dy = DELTA[move_angle]
            self.move_agent(dx, dy)

        elif action_str == "MoveBack":
            dx, dy = DELTA[round((move_angle + 180) % 360)]
            self.move_agent(dx, dy)

        elif action_str == "MoveLeft":
            dx, dy = DELTA[round((move_angle - 90) % 360)]
            self.move_agent(dx, dy)

        elif action_str == "MoveRight":
            dx, dy = DELTA[round((move_angle + 90) % 360)]
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

        info["state_value"] = self.get_state_value()

        return obs, reward, termination, truncation, info

    def get_state_value(self) -> float:
        # s = 0.0

        # for ent in self.entities:
        #     if isinstance(ent, MeshEnt):
        #         ent_x, ent_y, ent_z = ent.pos  # type: ignore
        #         agent_x, agent_y, agent_z = self.agent.pos  # type: ignore
        #         distance = math.sqrt(
        #             (ent_x - agent_x) ** 2
        #             + (ent_y - agent_y) ** 2
        #             + (ent_z - agent_z) ** 2
        #         )
        #         if distance < 3:
        #             s += 1
        # return s
        return len(self.get_visible_ents())

    def reset(self) -> tuple:
        super().reset()
        self.initialize_gt_values()

        # randomly teleport agent
        self.teleport_agent(random.choice(self.enumerate_poses()))
        self.step_count = 0

        info = {}
        info["state_value"] = self.get_state_value()
        return (None, info)
