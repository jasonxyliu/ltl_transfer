"""
Get robot pose in vision frame. Used to discretize environment.
"""

import bosdyn.client

if __name__ == '__main__':
    sdk = bosdyn.client.create_standard_sdk('show_robot_state')
    robot = sdk.create_robot('gouger')
    robot.authenticate('user', 'bigbubbabigbubba')
    state_client = robot.ensure_client('robot-state')
    transforms_snapshot = state_client.get_robot_state().kinematic_state.transforms_snapshot
    print(transforms_snapshot)
