'''
### Key Functions

- **Collision Detection** (`checkCollidesAt`):
    - Checks if the robot's collision volume is in a colliding state at a given position.
- **Path Visualization** (`visualizePath` and `visualizeCollisionFreeNodes`):
    - Draws the computed path and any collision-free nodes on the simulation interface for debugging and visualization.
- **Target Position Handling** (`getTargetPosition`):
    - Retrieves the current target position of the robot, incorporating some latency handling for stability.

### Main Control Logic (`coroutineMain`)

1. **Collision Handling**:
    - Ensures that the robot starts from a non-colliding position. If it does, it attempts to adjust its position until it is safe.
2. **Goal Management**:
    - Determines a goal position and checks its distance from the robot. If the goal is too far or colliding, it adjusts the goal position.
3. **Path Planning**:
    - If the goal is reachable, it creates a path using the OMPL (Open Motion Planning Library) and visualizes it.
    - It handles potential goal movements during planning, re-computing paths if necessary.
4. **Movement Execution**:
    - Once a path is found, the robot tracks the path by actuating its motors toward the target points.
    - Adjusts the velocities of the left and right wheels based on the desired direction.
5. **Stopping Conditions**:
    - Stops the robot when it reaches the goal or if the goal moves significantly.
'''
