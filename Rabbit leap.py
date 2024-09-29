from collections import deque

# Define the grid size and the jump distance
GRID_SIZE = 5
JUMP_DISTANCE = 2

# Function to check if the position is within grid bounds
def is_valid(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

# BFS implementation
def bfs(start, target):
    queue = deque([start])
    visited = set([start])
    distance = 0
    
    while queue:
        for _ in range(len(queue)):
            x, y = queue.popleft()
            if (x, y) == target:
                return distance
            
            for dx in range(-JUMP_DISTANCE, JUMP_DISTANCE + 1):
                for dy in range(-JUMP_DISTANCE, JUMP_DISTANCE + 1):
                    if abs(dx) + abs(dy) == JUMP_DISTANCE:
                        next_pos = (x + dx, y + dy)
                        if is_valid(next_pos[0], next_pos[1]) and next_pos not in visited:
                            visited.add(next_pos)
                            queue.append(next_pos)
        
        distance += 1
    return -1  # Target not reachable

# DFS implementation
def dfs(current, target, visited):
    if current == target:
        return 0
    
    visited.add(current)
    min_distance = float('inf')
    
    for dx in range(-JUMP_DISTANCE, JUMP_DISTANCE + 1):
        for dy in range(-JUMP_DISTANCE, JUMP_DISTANCE + 1):
            if abs(dx) + abs(dy) == JUMP_DISTANCE:
                next_pos = (current[0] + dx, current[1] + dy)
                if is_valid(next_pos[0], next_pos[1]) and next_pos not in visited:
                    distance = dfs(next_pos, target, visited)
                    if distance != -1:
                        min_distance = min(min_distance, distance + 1)
    
    visited.remove(current)
    return min_distance if min_distance != float('inf') else -1  # Target not reachable

# Main code to run both searches
def main():
    start = (0, 0)  # Starting position of the rabbit
    target = (4, 4)  # Target position
    
    # BFS
    bfs_result = bfs(start, target)
    print(f"BFS: The shortest path length is {bfs_result}")
    
    # DFS
    dfs_result = dfs(start, target, set())
    print(f"DFS: The shortest path length is {dfs_result}")

if __name__ == "__main__":
    main()
