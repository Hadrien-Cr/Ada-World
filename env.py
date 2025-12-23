from miniworld.entity import MeshEnt
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS
import math
from dataclasses import dataclass

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
GRID_STEP = 0.25
V_ANGLE_STEP = 45
H_ANGLE_STEP = 30
V_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
H_ANGLES = [-30, 0, 30]
V_ANGLES_INV = {angle: idx for idx, angle in enumerate(V_ANGLES)}
H_ANGLES_INV = {angle: idx for idx, angle in enumerate(H_ANGLES)}


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
    
    def discretize(self, grid_size):
        idx_x = int(self.x // grid_size)
        idx_z = int(self.z // grid_size)
        idx_yaw = V_ANGLES_INV[int(self.yaw % 360)]
        idx_pitch = H_ANGLES_INV[int(self.camera_horizon)]
        return (idx_x, idx_z, idx_yaw, idx_pitch)

def agent_pose_from_discrete(idx_x, idx_z, idx_yaw, idx_pitch, grid_size):
    x = idx_x * grid_size
    z = idx_z * grid_size
    yaw = V_ANGLES[idx_yaw]
    camera_horizon = H_ANGLES[idx_pitch]
    return AgentPose(x, 0, z, yaw, camera_horizon)


class ManyObjectsEnv(MiniWorldEnv):
    def __init__(self, n, grid_size, **kwargs):
        self.n = n
        assert grid_size >= 2
        self.grid_size = grid_size

        super().__init__(max_episode_steps=100_000, **kwargs)

    def _gen_world(self):

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

        # Place the agent a random distance away from the goal
        self.place_agent(
            pos = (self.grid_size // 2, 0, self.grid_size // 2),
            dir = 0
        )



    def move_agent(self, dx, dy) -> bool:
        self.agent.pos = (
            self.agent.pos
            + self.agent.dir_vec * dx
            + self.agent.right_vec * dy
        )
        return True

    def turn_agent(self, turn_angle) -> bool:
        turn_angle *= math.pi / 180
        orig_dir = self.agent.dir

        self.agent.dir += turn_angle
        self.agent.dir %= 2 * math.pi
        return True

    def pitch_agent(self, pitch_angle) -> bool:
        orig_pitch = self.agent.cam_pitch

        if self.agent.cam_pitch + pitch_angle > 30 or self.agent.cam_pitch + pitch_angle < -30:
            return False

        self.agent.cam_pitch += pitch_angle
        return True

    def enumerate_poses(self) -> list[AgentPose]:
        poses = []
        for ix in range(self.grid_size):
            for iz in range(self.grid_size):
                for iyaw in range(len(V_ANGLES)):
                    for ipitch in range(len(H_ANGLES)):
                        pose = agent_pose_from_discrete(ix, iz, iyaw, ipitch, 1)
                        poses.append(pose)
        return poses

    def teleport_agent(self, pose: AgentPose):
        self.agent.pos = (pose.x, pose.y, pose.z)
        self.agent.dir = pose.yaw * math.pi / 180
        self.agent.cam_pitch = pose.camera_horizon

    def step(self, action_str: str):
        self.step_count += 1

        if action_str == "MoveAhead":
            self.move_agent(GRID_STEP, 0)

        elif action_str == "MoveBack":
            self.move_agent(-GRID_STEP, 0)
        
        elif action_str == "MoveLeft":
            self.move_agent(0, -GRID_STEP)
        
        elif action_str == "MoveRight":
            self.move_agent(0, GRID_STEP)

        elif action_str == "RotateLeft":
            self.turn_agent(V_ANGLE_STEP)

        elif action_str == "RotateRight":
            self.turn_agent(-V_ANGLE_STEP)
        
        elif action_str == "LookUp":
            self.pitch_agent(H_ANGLE_STEP)
        
        elif action_str == "LookDown":
            self.pitch_agent(-H_ANGLE_STEP)

        obs = None

        reward = 0
        termination = False
        truncation = False
        info = {}
        
        info["visible_objects"] = len(self.get_visible_ents())
        print(f"Visible objects: {info['visible_objects']}")

        return obs, reward, termination, truncation, info