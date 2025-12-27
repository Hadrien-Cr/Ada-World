import heapq
from env import DiscretizedAgentPose


def check_symmetry(w: dict[int, list[tuple[int, float, int]]]) -> bool:
    for u, neighbors in w.items():
        for action, cost, v in neighbors:
            if v not in w:
                print(f"Asymmetry: {v} is a neighbor of {u} but not a source.")
                return False

            found = False
            for _, r_cost, r_v in w[v]:
                if r_v == u:
                    if abs(r_cost - cost) > 1e-6:
                        print(
                            f"Cost Asymmetry: {u}->{v} is {cost}, but {v}->{u} is {r_cost}"
                        )
                        return False
                    found = True
                    break

            if not found:
                print(
                    f"Structural Asymmetry: {u}->{v} exists, but no edge exists from {v}->{u}"
                )
                return False
    return True


def get_weighted_adjacency_list(
    values: dict[DiscretizedAgentPose, float],
    adjacency_list: dict[DiscretizedAgentPose, list[tuple[int, DiscretizedAgentPose]]],
    visited_positions: set[DiscretizedAgentPose],
) -> dict[DiscretizedAgentPose, list[tuple[int, float, DiscretizedAgentPose]]]:
    weighted_adjacency_list = {}

    for node, neighbors in adjacency_list.items():
        weighted_adjacency_list[node] = []

        for action, n_node_tuple in neighbors:
            value_n = (values[n_node_tuple] + values[node]) / 2
            assert 0.0 <= value_n <= 1.0
            cost = (
                2.0 - value_n
                if (node in visited_positions and n_node_tuple in visited_positions)
                else 2.0
            )
            weighted_adjacency_list[node].append((action, cost, n_node_tuple))

    assert check_symmetry(weighted_adjacency_list), "Adjacency list is not symmetric"
    return weighted_adjacency_list


def astar(
    start: int,
    goal: int,
    w: dict[int, list[tuple[int, float, int]]],
    int2pose: dict[int, DiscretizedAgentPose],
) -> list[tuple[int, int]]:
    """
    pose2coords: dict mapping node ID to (x, y) tuple
    """
    # Priority Queue stores (f_score, current_cost, node)
    # f_score = current_cost + heuristic
    pq = [(0.0, 0.0, start)]

    distances = {start: 0.0}
    parent_map = {start: (None, None)}

    goal_pose = int2pose[goal]

    while pq:
        f_score, current_cost, u = heapq.heappop(pq)

        if u == goal:
            return reconstruct_path(parent_map, goal)

        if current_cost > distances.get(u, float("inf")):
            continue

        for action, edge_cost, v in w.get(u, []):
            new_cost = current_cost + edge_cost

            if v not in distances or new_cost < distances[v]:
                distances[v] = new_cost
                parent_map[v] = (u, action)  # type: ignore

                # HEURISTIC CALCULATION
                vpose = int2pose[v]
                h = (
                    abs(vpose.idx_x - goal_pose.idx_x)
                    + abs(vpose.idx_z - goal_pose.idx_z)
                    + abs(((vpose.idx_yaw - goal_pose.idx_yaw) % 8 - 4))
                    + abs((vpose.idx_h - goal_pose.idx_h))
                )

                # If your min edge weight is ~1.0, h * 1.0 is safe.
                # If your min edge weight is 0.5, use h * 0.5.
                f_score = new_cost + 1.0 * h

                heapq.heappush(pq, (f_score, new_cost, v))

    return []


def reconstruct_path(parent_map, goal):
    path = []
    curr = goal
    while curr is not None:
        parent, action = parent_map[curr]
        if action is not None:
            path.append((action, curr))
        curr = parent
    return path[::-1]
