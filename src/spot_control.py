import sys
import argparse
import time
import math

import bosdyn.client
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b
from bosdyn.client.docking import blocking_undock, blocking_dock_robot
from bosdyn.client import math_helpers
from bosdyn.api import robot_command_pb2
from bosdyn.api import geometry_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pd2
from bosdyn.util import seconds_to_duration

from demo_env import COORD2POS


def spot_execute_option(cur_loc, actions):


    # Initialize a robot command message, which we will build out below
    command = robot_command_pb2.RobotCommand()


def move_base(config):
    """
    Move robot base to a give pose in space
    """
    # Initialize robot
    sdk = create_standard_sdk("move_robot_base")
    robot = sdk.create_robot(config.hostname)

    robot.authenticate(username=config.username, password=config.password)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        if robot_state_client.get_robot_state().power_state.shore_power_state == 1:
            # Undock if docked
            robot.logger.info("Robot undocking...\nCLEAR AREA in front of docking station.")
            blocking_undock(robot)
            robot.logger.info("Robot undocked and standing")
            time.sleep(3)
        else:
            # Stand if not docked
            robot.logger.info("Commanding robot to stand...")
            blocking_stand(robot_command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")
            time.sleep(3)

        # Move to a given xy position
        # transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        # vision_T_body = get_vision_tform_body(transforms)

        # Initialize a robot command message, which we will build out below
        command = robot_command_pb2.RobotCommand()

        # # Walk to origin
        # point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
        # pos_vision, rot_vision = COORD2POS[(2, 0, 0)]  # pose relative to vision frame
        # point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
        # point.pose.angle = yaw_angle(rot_vision)
        # point.time_since_reference.CopyFrom(seconds_to_duration(25))

        # Walk through a sequence of coordinates
        coords = plan_trajectory(config.grid_x, config.grid_y)
        for idx, coord in enumerate(coords):
            print(f"adding coordinate to command: {coord}")
            point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
            pos_vision, rot_vision = COORD2POS[coord]  # pose relative to vision frame
            point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
            point.pose.angle = yaw_angle(rot_vision)
            traj_time = (idx + 1) * config.time_per_move
            point.time_since_reference.CopyFrom(seconds_to_duration(traj_time))

        command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = VISION_FRAME_NAME

        # speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=2, y=2), angular=0),
        #                                min_vel=SE2Velocity(linear=Vec2(x=-2, y=-2), angular=0))
        # mobility_command = spot_command_pd2.MobilityParams(vel_limit=speed_limit)
        # command.synchronized_command.mobility_command.params.CopyFrom(RobotCommandBuilder._to_any(mobility_command))

        # Send the command using command client
        time_full = config.time_per_move * len(coords)
        robot.logger.info("Send body trajectory command.")
        robot_command_client.robot_command(command, end_time_secs=time.time() + time_full)
        time.sleep(time_full + 2)

        if config.dock_after_use:
            # Dock robot after mission complete
            blocking_dock_robot(robot, config.dock_id)
            robot.logger.info("Robot docked")

        if config.poweroff_after_dock:
            # Power off
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed"
            robot.logger.info("Robot safely powered off")


def plan_trajectory(dest_x, dest_y):
    """
    Plan robot trajectory to destination position in grid
    """
    return [(0, 0, 0)]
    # return [(2, 0, 1), (2, 1, 1), (2, 1, 2), (0.2, 1, 2), (0.2, 1, 0),
    #         (2, 1, 0), (3, 1, 0), (6, 1, 0), (6, 1, 1), (6, 4, 1), (6, 4, 2),
    #         (5, 4, 2), (5, 4, 1), (5, 5, 1), (5, 11, 2), (5, 11, 3), (5, 5, 3), (5, 0, 2), (2, 0, 2), (2, 0, 0)
    #         ]  # go to book shelf, desk, kitchen then back
    # return [(3, 0, 0), (5, 0, 0), (5, 5, 1), (5, 11, 2), (5, 11, 3), (5, 5, 3), (5, 0, 2), (2, 0, 2), (2, 0, 0)]  # go to kitchen then back


def move_arm(config):
    """
    Move robot arm to a given pose in space
    """
    # Initialize robot
    sdk = create_standard_sdk("move_robot_arm")
    robot = sdk.create_robot(config.hostname)

    robot.authenticate(username=config.username, password=config.password)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "Robot requires an arm to run this program"
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed"
        robot.logger.info("Robot powered on.")

        # # Undock
        # robot.logger.info("Robot undocking...\nCLEAR AREA in front of docking station.")
        # blocking_undock(robot)
        # robot.logger.info("Robot undocked and standing")
        # time.sleep(3)

        # # Stand
        # robot.logger.info("Commanding robot to stand...")
        # blocking_stand(command_client, timeout_sec=10)
        # robot.logger.info("Robot standing.")
        # time.sleep(3)

        # # Stow arm
        # stow_command = RobotCommandBuilder.arm_stow_command()
        # stow_command_id = command_client.robot_command(stow_command)
        # robot.logger.info("Stow command issued")
        # block_until_arm_arrives(command_client, stow_command_id, 3.0)
        # time.sleep(3)

        # Move arm to a given pose


        # Initialize a robot command message, which we will build out below
        command = robot_command_pb2.RobotCommand()


        # # Unstow arm
        # unstow_command = RobotCommandBuilder.arm_ready_command()
        # unstow_command_id = command_client.robot_command(unstow_command)
        # robot.logger.info("Unstow command issued.")
        # block_until_arm_arrives(command_client, unstow_command_id, 3.0)
        # time.sleep(3)

        # Dock robot after mission complete
        blocking_dock_robot(robot, config.dock_id)
        robot.logger.info("Robot docked.")

        # # Power off
        # robot.power_off(cut_immediately=False, timeout_sec=20)
        # assert not robot.is_powered_on(), "Robot power off failed"
        # robot.logger.info("Robot safely powered off")


def yaw_angle(rot_vision):
    if len(rot_vision) == 3:  # euler angel
        return deg2rad(rot_vision[2])  # only yaw
    else:  # quaternion
        return math_helpers.quat_to_eulerZYX(math_helpers.Quat(*rot_vision))[0]  # only yaw


def deg2rad(deg):
    """
    Convert degrees to radians
    """
    return deg / 180.0 * math.pi


def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--username", type=str, default="user", help="Username of Spot")
    parser.add_argument("--password", type=str, default="97qp5bwpwf2c", help="Password of Spot")  # dungnydsc8su
    parser.add_argument("--dock_id", required=True, type=int, help="Docking station ID to dock at")
    parser.add_argument("---grid_x", type=int, default=1, help="X coordinate of grid cell to move robot base to")
    parser.add_argument("--grid_y", type=int, default=0, help="Y coordinate of grid cell to move robot base to")
    parser.add_argument("--time_per_move", type=int, default=25, help="Seconds each move in grid should take")
    parser.add_argument('--dock_after_use', action="store_true", help='Include to dock Spot after operation')
    parser.add_argument('--poweroff_after_use', action="store_true", help='Include to power off Spot after operation')
    config = parser.parse_args(argv)
    bosdyn.client.util.setup_logging(config.verbose)

    try:
        move_base(config)
        # move_arm(config)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.error(f"Robot control threw an exception: {exc}")
        return False


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
