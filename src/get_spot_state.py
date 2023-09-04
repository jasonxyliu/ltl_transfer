"""
Get robot pose in vision frame. Used to discretize environment.
"""
import argparse
import bosdyn.client

if __name__ == '__main__':
    # python3 src/get_spot_state.py --robot_name=tusker
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot to operate.")
    args = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk('show_robot_state')
    robot = sdk.create_robot(args.robot_name)
    robot.authenticate('user', 'bigbubbabigbubba')
    state_client = robot.ensure_client('robot-state')
    transforms_snapshot = state_client.get_robot_state().kinematic_state.transforms_snapshot
    print(transforms_snapshot)
