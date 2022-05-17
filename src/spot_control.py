import sys
import argparse
import time
import math

from game_objects import Actions

import bosdyn.client
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.docking import blocking_undock, blocking_dock_robot
from bosdyn.client import math_helpers
from bosdyn.api import robot_command_pb2, geometry_pb2, arm_command_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pd2
from bosdyn.util import seconds_to_duration

from env_map import COORD2LOC, CODE2ROT, COORD2GPOSE


def spot_execute_action(robot, config, robot_command_client, cur_loc, action):
    # Initialize a robot command message, which we will build out below
    command = robot_command_pb2.RobotCommand()

    # Walk through a sequence of coordinates
    pose = action2pose(cur_loc, action)
    print(f"adding pose to command: {pose}")
    point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
    loc_vision, rot_vision = COORD2LOC[pose[:2]], CODE2ROT[pose[2]]  # pose relative to vision frame
    point.pose.position.x, point.pose.position.y = loc_vision  # only x, y
    point.pose.angle = yaw_angle(rot_vision)
    point.time_since_reference.CopyFrom(seconds_to_duration(config.time_per_move))

    # Support frame
    command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = VISION_FRAME_NAME

    # speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=2, y=2), angular=0),
    #                                min_vel=SE2Velocity(linear=Vec2(x=-2, y=-2), angular=0))
    # mobility_command = spot_command_pd2.MobilityParams(vel_limit=speed_limit)
    # command.synchronized_command.mobility_command.params.CopyFrom(RobotCommandBuilder._to_any(mobility_command))

    # Send the command using command client
    robot.logger.info("Send body trajectory command.")
    robot_command_client.robot_command(command, end_time_secs=time.time() + config.time_per_move)
    time.sleep(config.time_per_move + 2)


def action2pose(cur_loc, action):
    next_x, next_y = cur_loc

    if action == Actions.up:
        next_x -= 1
        rot_code = 0
    if action == Actions.down:
        next_x += 1
        rot_code = 2
    if action == Actions.left:
        next_y -= 1
        rot_code = 3
    if action == Actions.right:
        next_y += 1
        rot_code = 1

    # if (next_x, next_y) == (5, 3):  # always facing desk_a if it is the desination
    #     rot_code = 1

    if (next_x, next_y) == (1, 4):  # always facing desk_b if it is the desination
        rot_code = 1

    if (next_x, next_y) == (3, 10):  # always facing counter in kitchen
        rot_code = 2

    return next_x, next_y, rot_code


def spot_execute_option(robot, config, robot_command_client, cur_loc, actions):
    # Initialize a robot command message, which we will build out below
    command = robot_command_pb2.RobotCommand()

    # Walk through a sequence of coordinates
    poses = plan_trajectory(cur_loc, actions)
    for idx, pose in enumerate(poses):
        print(f"adding pose to command: {pose}")
        point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
        loc_vision, rot_vision = COORD2LOC[pose[:2]], CODE2ROT[pose[2]]  # pose relative to vision frame
        point.pose.position.x, point.pose.position.y = loc_vision  # only x, y
        point.pose.angle = yaw_angle(rot_vision)
        traj_time = (idx + 1) * config.time_per_move
        point.time_since_reference.CopyFrom(seconds_to_duration(traj_time))

    # Support frame
    command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = VISION_FRAME_NAME

    # speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=2, y=2), angular=0),
    #                                min_vel=SE2Velocity(linear=Vec2(x=-2, y=-2), angular=0))
    # mobility_command = spot_command_pd2.MobilityParams(vel_limit=speed_limit)
    # command.synchronized_command.mobility_command.params.CopyFrom(RobotCommandBuilder._to_any(mobility_command))

    # Send the command using command client
    time_full = config.time_per_move * len(poses)
    robot.logger.info("Send body trajectory command.")
    robot_command_client.robot_command(command, end_time_secs=time.time() + time_full)
    time.sleep(time_full + 2)


def plan_trajectory(cur_loc, actions):
    """
    Plan robot trajectory to destination pose in grid
    """
    next_x, next_y = cur_loc
    traj = []

    for action in actions:
        if action == Actions.up:
            next_x -= 1
            rot_code = 0
        if action == Actions.down:
            next_x += 1
            rot_code = 2
        if action == Actions.left:
            next_y -= 1
            rot_code = 3
        if action == Actions.right:
            next_y += 1
            rot_code = 1

        # if (next_x, next_y) == (5, 3):  # always facing desk_a if it is the desination
        #     rot_code = 1

        if (next_x, next_y) == (1, 4):  # always facing desk_b if it is the desination
            rot_code = 1

        if (next_x, next_y) == (3, 10):  # always facing counter in kitchen
            rot_code = 2

        traj.append((next_x, next_y, rot_code))

    return traj


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

        # Walk to origin
        poses = [(3, 3, 0)]
        point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
        pos_vision, rot_vision = COORD2LOC[poses[0][:2]], CODE2ROT[poses[0][2]]  # pose relative to vision frame
        point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
        point.pose.angle = yaw_angle(rot_vision)
        point.time_since_reference.CopyFrom(seconds_to_duration(config.time_per_move))

        # Walk through a sequence of coordinates
        cur_loc = (3, 3)
        actions = [Actions.left, Actions.left, Actions.down, Actions.down, Actions.down]
        poses = plan_trajectory(cur_loc, actions)
        for idx, pose in enumerate(poses):
            print(f"adding pose to command: {pose}")
            point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
            loc_vision, rot_vision = COORD2LOC[pose[:2]], CODE2ROT[pose[2]]  # pose relative to vision frame
            point.pose.position.x, point.pose.position.y = loc_vision  # only x, y
            point.pose.angle = yaw_angle(rot_vision)
            traj_time = (idx + 1) * config.time_per_move
            point.time_since_reference.CopyFrom(seconds_to_duration(traj_time))

        # Support frame
        command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = VISION_FRAME_NAME

        # speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=2, y=2), angular=0),
        #                                min_vel=SE2Velocity(linear=Vec2(x=-2, y=-2), angular=0))
        # mobility_command = spot_command_pd2.MobilityParams(vel_limit=speed_limit)
        # command.synchronized_command.mobility_command.params.CopyFrom(RobotCommandBuilder._to_any(mobility_command))

        # Send the command using command client
        time_full = config.time_per_move * len(poses)
        robot.logger.info("Send body trajectory command.")
        robot_command_client.robot_command(command, end_time_secs=time.time() + time_full)
        time.sleep(time_full + 2)

        if config.dock_after_use:
            # Dock robot after mission complete
            blocking_dock_robot(robot, config.dock_id)
            robot.logger.info("Robot docked")

        if config.poweroff_after_use:
            # Power off
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed"
            robot.logger.info("Robot safely powered off")


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
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed"
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

        # # Unstow arm
        # if robot_state_client.get_robot_state().manipulator_state.stow_state == 1:
        #     unstow_command = RobotCommandBuilder.arm_ready_command()
        #     unstow_command_id = robot_command_client.robot_command(unstow_command)
        #     robot.logger.info("Unstow command issued.")
        #     block_until_arm_arrives(robot_command_client, unstow_command_id, 3.0)
        #     time.sleep(3)

        move_gripper(COORD2GPOSE[6, 1], robot, robot_state_client, robot_command_client)

        # Stow arm
        if robot_state_client.get_robot_state().manipulator_state.stow_state == 2:
            stow_command = RobotCommandBuilder.arm_stow_command()
            stow_command_id = robot_command_client.robot_command(stow_command)
            robot.logger.info("Stow command issued")
            block_until_arm_arrives(robot_command_client, stow_command_id, 3.0)
            time.sleep(3)

        if config.dock_after_use:
            # Dock robot after mission complete
            blocking_dock_robot(robot, config.dock_id)
            robot.logger.info("Robot docked")

        if config.poweroff_after_use:
            # Power off
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed"
            robot.logger.info("Robot safely powered off")


def move_gripper(gripper_pose, robot, robot_state_client, robot_command_client):
    """
    Move arm to a spot in front of robot, then open gripper
    """
    (x, y, z), (qw, qx, qy, qz) = gripper_pose
    # Make arm pose RobotCommand
    # Build a position to move arm to (in meters, relative to and expressed in gravity aligned body frame)
    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    # Rotation as a quaternion
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
    # Build SE3Pose proto object
    flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body, rotation=flat_body_Q_hand)

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)
    print(f"odom_T_hand: {odom_T_hand}\n{type(odom_T_hand)}")

    duration_seconds = 2
    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x, odom_T_hand.y, odom_T_hand.z,
        odom_T_hand.rot.w, odom_T_hand.rot.x, odom_T_hand.rot.y, odom_T_hand.rot.z,
        ODOM_FRAME_NAME, duration_seconds)

    # Make open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

    # Combine arm and gripper commands into 1 RobotCommand
    robot_command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

    # Send request
    cmd_id = robot_command_client.robot_command(robot_command)
    robot.logger.info("Move arm to pose.")
    block_until_arm_arrives_with_prints(robot, robot_command_client, cmd_id)
    block_until_arm_arrives(robot_command_client, cmd_id)
    time.sleep(10)
    robot.logger.info("Arm move done.")


def block_until_arm_arrives_with_prints(robot, command_client, cmd_id):
    """
    Block until arm arrives at goal and print remaining distance of position and rotation.
    Note: a version w/o prints of this function is available as a helper in robot_command
    """
    while True:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        robot.logger.info(
            "Distance to go: " +
            f"{feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_pos_distance_to_goal:.2f} meters" +
            f"{feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_rot_distance_to_goal:.2f}"
        )

        if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            robot.logger.info("Arm reached goal")
            break
        time.sleep(0.1)


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
    parser.add_argument("--move", required=True, type=str, help="Move base or arm")
    parser.add_argument("--time_per_move", type=int, default=25, help="Seconds each move in grid should take")
    parser.add_argument('--dock_after_use', action="store_true", help='Include to dock Spot after operation')
    parser.add_argument('--poweroff_after_use', action="store_true", help='Include to power off Spot after operation')
    config = parser.parse_args(argv)
    bosdyn.client.util.setup_logging(config.verbose)

    try:
        if config.move == "base":
            move_base(config)
        else:
            move_arm(config)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.error(f"Robot control threw an exception: {exc}")
        return False


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
