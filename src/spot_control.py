import sys
import os
import argparse
import time
import math
import cv2
import numpy as np
import tensorflow as tf

import bosdyn.client
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.docking import blocking_undock, blocking_dock_robot
from bosdyn.client import math_helpers
from bosdyn.api import robot_command_pb2, geometry_pb2, arm_command_pb2, manipulation_api_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pd2
from bosdyn.util import seconds_to_duration

from network_compute_server import TensorFlowObjectDetectionModel
from game_objects import Actions
from env_map import COORD2LOC, CODE2ROT, COORD2GPOSE, PICK_PROPS, PICK_PROPS_INV, PLACE_PROPS, PLACE_PROPS_INV


MODEL2PATHS = {
    "book_pr": ("multiobj/exported_models/model_book_pr_hand_gray/saved_model",
                "multiobj/annotations/label_map.pbtxt"),
    "juice": ("multiobj/exported_models/juice_orange_hand_color/saved_model",
              "multiobj/annotations_juice_hand_color/label_map.pbtxt")
}

COORD2MODE = {
    (6, 1): "book_pr",
    (3, 10): "juice"
}


def navigate(robot, config, robot_command_client, cur_loc, action, goal_prop):
    # Initialize a robot command message, which we will build out below
    command = robot_command_pb2.RobotCommand()

    # Walk through a sequence of coordinates
    pose = action2pose(cur_loc, action, goal_prop)
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

    return pose


def action2pose(cur_loc, action, goal_prop):
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

    if goal_prop == 'a' and (next_x, next_y) == PLACE_PROPS['a']:  # always facing desk_a if it is the destination
        rot_code = 1

    if (next_x, next_y) == (1, 4):  # always facing desk_b if it is the destination
        rot_code = 1

    if (next_x, next_y) == (3, 10):  # always facing counter in kitchen
        rot_code = 2

    return next_x, next_y, rot_code


def navigate_seq(robot, config, robot_command_client, cur_loc, actions):
    """
    Use navigate
    """
    # Initialize a robot command message, which we will build out below
    command = robot_command_pb2.RobotCommand()

    # Walk through a sequence of coordinates
    poses, _ = plan_trajectory(cur_loc, actions)
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
    time.sleep(time_full + 1)


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

    return traj, (next_x, next_y)


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
            time.sleep(1)
        else:
            # Stand if not docked
            robot.logger.info("Commanding robot to stand...")
            blocking_stand(robot_command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")
            time.sleep(1)

        # Move to a given xy position
        # transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        # vision_T_body = get_vision_tform_body(transforms)

        # Initialize a robot command message, which we will build out below
        command = robot_command_pb2.RobotCommand()

        # # Walk to origin
        # poses = [(3, 3, 0)]
        # point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
        # pos_vision, rot_vision = COORD2LOC[poses[0][:2]], CODE2ROT[poses[0][2]]  # pose relative to vision frame
        # point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
        # point.pose.angle = yaw_angle(rot_vision)
        # point.time_since_reference.CopyFrom(seconds_to_duration(config.time_per_move))
        # # Support frame
        # command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = VISION_FRAME_NAME
        # # speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=2, y=2), angular=0),
        # #                                min_vel=SE2Velocity(linear=Vec2(x=-2, y=-2), angular=0))
        # # mobility_command = spot_command_pd2.MobilityParams(vel_limit=speed_limit)
        # # command.synchronized_command.mobility_command.params.CopyFrom(RobotCommandBuilder._to_any(mobility_command))
        # # Send the command using command client
        # time_full = config.time_per_move * len(poses)
        # robot.logger.info("Send body trajectory command.")
        # robot_command_client.robot_command(command, end_time_secs=time.time() + time_full)
        # time.sleep(time_full + 1)

        # cur_loc = poses[0][:2]
        # actions = [Actions.down, Actions.down]
        # goal_prop = PLACE_PROPS_INV[plan_trajectory(cur_loc, actions)[1]]
        # for action in actions:
        #     cur_pose = navigate(robot, config, robot_command_client, cur_loc, action, goal_prop)
        #     cur_loc = cur_pose[:2]

        if config.dock_after_use:
            # Initialize a robot command message, which we will build out below
            command = robot_command_pb2.RobotCommand()
            # Walk to initial location
            poses = [(0, 0, 0)]
            point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
            pos_vision, rot_vision = COORD2LOC[poses[0][:2]], CODE2ROT[poses[0][2]]  # pose relative to vision frame
            point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
            point.pose.angle = yaw_angle(rot_vision)
            point.time_since_reference.CopyFrom(seconds_to_duration(config.time_per_move))
            # Support frame
            command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = VISION_FRAME_NAME
            # Send the command using command client
            robot.logger.info("Send body trajectory command.")
            robot_command_client.robot_command(command, end_time_secs=time.time() + config.time_per_move)
            time.sleep(config.time_per_move + 1)

            # Dock robot after mission complete
            blocking_dock_robot(robot, config.dock_id)
            robot.logger.info("Robot docked")

        if config.poweroff_after_use:
            # Power off
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed"
            robot.logger.info("Robot safely powered off")


def nav_grasp(config):
    # Initialize robot
    sdk = create_standard_sdk("move_robot_arm")
    robot = sdk.create_robot(config.hostname)

    robot.authenticate(username=config.username, password=config.password)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "Robot requires an arm to run this program"
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_image_client = robot.ensure_client(ImageClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # Pre-load detector models
    models = {}
    for model_name in MODEL2PATHS.keys():
        model_dpath, label_fpath = MODEL2PATHS[model_name]
        model = TensorFlowObjectDetectionModel(model_dpath, label_fpath)
        models[model_name] = model

        # Preload tf model
        image_responses = robot_image_client.get_image_from_sources(['hand_color_image'])
        # Unpack image
        image = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
        image = cv2.imdecode(image, -1)
        model.predict(image)
        print(f"loaded model {model_dpath}")

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
            time.sleep(1)
        else:
            # Stand if not docked
            robot.logger.info("Commanding robot to stand...")
            blocking_stand(robot_command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")
            time.sleep(1)

        # Initialize a robot command message, which we will build out below
        command = robot_command_pb2.RobotCommand()

        # Walk to origin
        poses = [(3, 3, 0)]
        point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
        pos_vision, rot_vision = COORD2LOC[poses[0][:2]], CODE2ROT[poses[0][2]]  # pose relative to vision frame
        point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
        point.pose.angle = yaw_angle(rot_vision)
        point.time_since_reference.CopyFrom(seconds_to_duration(config.time_per_move))
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
        time.sleep(time_full + 1)

        cur_loc = poses[0][:2]
        actions = [Actions.down, Actions.down, Actions.place]

        final_loc = plan_trajectory(cur_loc, actions)[1]
        goal_prop = None
        if final_loc in PLACE_PROPS_INV:
            goal_prop = PLACE_PROPS_INV[final_loc]
        if final_loc in PICK_PROPS_INV:
            goal_prop = PICK_PROPS_INV[final_loc]

        # actions = [Actions.right] * 7 + [Actions.pick] + [Actions.left] * 7 + [Actions.down, Actions.down, Actions.place]
        # actions = [
        #            Actions.down, Actions.down,   # to a
        #            Actions.pick, Actions.up, Actions.right, Actions.right, Actions.place,  # to a
        #            Actions.pick,
        #            ]
        for action in actions:
            if action == Actions.pick:
                pick(config, robot, robot_state_client, robot_command_client, robot_image_client, robot_manipulation_client, models["book_pr"], cur_loc)
            elif action == Actions.place:
                place(robot, robot_state_client, robot_command_client, cur_loc, 3)
            elif action == "capture_image":
                move_gripper(robot, robot_state_client, robot_command_client, COORD2GPOSE[cur_loc[0], cur_loc[1]], 1.0, 10, True)
            else:
                cur_pose = navigate(robot, config, robot_command_client, cur_loc, action, goal_prop)
                cur_loc = cur_pose[:2]


def test(config):
    # Initialize robot
    sdk = create_standard_sdk("move_robot_arm")
    robot = sdk.create_robot(config.hostname)

    robot.authenticate(username=config.username, password=config.password)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), "Robot requires an arm to run this program"
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_image_client = robot.ensure_client(ImageClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # Pre-load detector models
    models = {}
    for model_name in MODEL2PATHS.keys():
        model_dpath, label_fpath = MODEL2PATHS[model_name]
        model = TensorFlowObjectDetectionModel(model_dpath, label_fpath)
        models[model_name] = model

        # Preload tf model
        image_responses = robot_image_client.get_image_from_sources(['hand_color_image'])
        # Unpack image
        image = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
        image = cv2.imdecode(image, -1)
        model.predict(image)
        print(f"loaded model {model_dpath}")

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
            time.sleep(1)
        else:
            # Stand if not docked
            robot.logger.info("Commanding robot to stand...")
            blocking_stand(robot_command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")
            time.sleep(1)

        # Initialize a robot command message, which we will build out below
        command = robot_command_pb2.RobotCommand()

        # Walk to origin
        poses = [(3, 3, 0)]
        point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add()
        pos_vision, rot_vision = COORD2LOC[poses[0][:2]], CODE2ROT[poses[0][2]]  # pose relative to vision frame
        point.pose.position.x, point.pose.position.y = pos_vision[0], pos_vision[1]  # only x, y
        point.pose.angle = yaw_angle(rot_vision)
        point.time_since_reference.CopyFrom(seconds_to_duration(config.time_per_move))
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
        time.sleep(time_full + 1)

        cur_loc = poses[0][:2]
        options = [
            [Actions.left] * 2 + [Actions.down] * 3,
            [Actions.up, Actions.right, Actions.right],
            [Actions.up] * 2 + [Actions.right] * 7,
            [Actions.left] * 7 + [Actions.down] * 2,
        ]
        for option in options:
            cur_loc = spot_execute_actions(config, robot, robot_state_client, robot_command_client, robot_image_client, robot_manipulation_client,
                                           models, cur_loc, option)


def spot_execute_actions(config, robot, robot_state_client, robot_command_client, robot_image_client, robot_manipulation_client,
                         models, cur_loc, actions):
    print(f"EXECUTE option: {actions}")

    final_loc = plan_trajectory(cur_loc, actions)[1]
    goal_prop = None
    if final_loc in PLACE_PROPS_INV:
        goal_prop = PLACE_PROPS_INV[final_loc]
    if final_loc in PICK_PROPS_INV:
        goal_prop = PICK_PROPS_INV[final_loc]

    for action in actions:
        cur_loc = spot_execute_action(config, robot, robot_state_client, robot_command_client, robot_image_client, robot_manipulation_client,
                                      models, cur_loc, action, goal_prop)
    return cur_loc


def spot_execute_action(config, robot, robot_state_client, robot_command_client, robot_image_client, robot_manipulation_client,
                        models, cur_loc, action, goal_prop):
    next_pose = navigate(robot, config, robot_command_client, cur_loc, action, goal_prop)
    cur_loc = next_pose[:2]

    if goal_prop in PICK_PROPS.keys() and cur_loc == PICK_PROPS[goal_prop]:
        pick(config, robot, robot_state_client, robot_command_client, robot_image_client, robot_manipulation_client, models[COORD2MODE[cur_loc]], cur_loc)
    if goal_prop in PLACE_PROPS.keys() and cur_loc == PLACE_PROPS[goal_prop]:
        place(robot, robot_state_client, robot_command_client, cur_loc, 3)
    return cur_loc


def pick(config, robot, robot_state_client, robot_command_client, robot_image_client, robot_manipulation_client, model, coord):
    box2conf, image_resps = get_boxes(config, robot, robot_state_client, robot_command_client, robot_image_client, COORD2GPOSE[coord[0], coord[1]], 25, model)

    # Find overlapping region
    box, conf = sorted(box2conf.items(), key=lambda kv: kv[1])[-1]
    image_resp = image_resps[list(box2conf.keys()).index(box)]
    print(f"most confident box: {box}, {conf}")

    # Pick bounding box center
    center_x, center_y = find_center_px(box)

    # Stow
    stow_command = RobotCommandBuilder.arm_stow_command()
    stow_command_id = robot_command_client.robot_command(stow_command)
    robot.logger.info("Stow command issued")
    block_until_arm_arrives(robot_command_client, stow_command_id, 3.0)
    time.sleep(1)

    # Grasp
    pick_vec = geometry_pb2.Vec2(x=center_x, y=center_y)
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec, transforms_snapshot_for_camera=image_resp.shot.transforms_snapshot,
        frame_name_image_sensor=image_resp.shot.frame_name_image_sensor,
        camera_model=image_resp.source.pinhole)
    grasp.grasp_params.grasp_palm_to_fingertip = 0
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
    # Build proto
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
    # Send the request
    cmd_response = robot_manipulation_client.manipulation_api_command(manipulation_api_request=grasp_request)
    # Get feedback from the robot
    moving_counter = 0
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_response.manipulation_cmd_id)
        # Send the request
        response = robot_manipulation_client.manipulation_api_feedback_command(manipulation_api_feedback_request=feedback_request)
        print('Current state: ', manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))
        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or \
            moving_counter > 20 or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            break
        if response.current_state == manipulation_api_pb2.MANIP_STATE_MOVING_TO_GRASP:
            moving_counter += 1
        time.sleep(0.25)

    # Carry
    carry_command = RobotCommandBuilder.arm_carry_command()
    carry_command_id = robot_command_client.robot_command(carry_command)
    robot.logger.info("Carry command issued")
    block_until_arm_arrives(robot_command_client, carry_command_id, 3.0)
    time.sleep(1)


def place(robot, robot_state_client, robot_command_client, coord, hold_time):
    arm_command = move_gripper(robot, robot_state_client, robot_command_client, COORD2GPOSE[coord[0], coord[1]], 0, 1)
    arm_command = move_gripper(robot, robot_state_client, robot_command_client, COORD2GPOSE[coord[0], coord[1]], 1, hold_time, arm_command)
    move_gripper(robot, robot_state_client, robot_command_client, COORD2GPOSE[coord[0], coord[1]], 0, hold_time, arm_command)
    # Stow
    stow_command = RobotCommandBuilder.arm_stow_command()
    stow_command_id = robot_command_client.robot_command(stow_command)
    robot.logger.info("Stow command issued")
    block_until_arm_arrives(robot_command_client, stow_command_id, 3.0)
    time.sleep(1)


def get_boxes(config, robot, robot_state_client, robot_command_client, robot_image_client, gripper_pose, hold_time, model):
    """
    Move arm to a spot in front of robot, open gripper, then keep taking images until bounding box is detected
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

    # Send request and hold pose for 'hold_time'
    cmd_id = robot_command_client.robot_command(robot_command)
    robot.logger.info("Move arm to pose.")
    block_until_arm_arrives_with_prints(robot, robot_command_client, cmd_id)
    # block_until_arm_arrives(robot_command_client, cmd_id)

    hold_until_time = time.time() + hold_time
    box2conf = {}
    image_resps = []
    counter = 0

    while time.time() < hold_until_time:
        print(f"Counter: {counter}")
        # Take image
        image_responses = robot_image_client.get_image_from_sources(['hand_color_image'])
        # Unpack image
        image = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
        image = cv2.imdecode(image, -1)
        image_width, image_height = image.shape[0], image.shape[1]

        # Inference model to get bounding box
        detections = model.predict(image)

        # print(detections)
        print(f"num of detections: ", detections["num_detections"])

        num_detections = int(detections.pop('num_detections'))
        if num_detections > 0:
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            boxes = detections['detection_boxes']
            classes = detections['detection_classes']
            scores = detections['detection_scores']

            print(f"num of boxes: {len(boxes)}")

            label = classes[0]
            score = scores[0]

            print(label, score)

            box = tuple(boxes[0].tolist())
            box = [box[0] * image_width, box[1] * image_height, box[2] * image_width, box[3] * image_height]

            point1 = np.array([box[1], box[0]])
            point2 = np.array([box[3], box[0]])
            point3 = np.array([box[3], box[2]])
            point4 = np.array([box[1], box[2]])

            vertex1 = (point1[0], point1[1])
            vertex2 = (point2[0], point2[1])
            vertex3 = (point3[0], point3[1])
            vertex4 = (point4[0], point4[1])

            box2conf[(vertex1, vertex2, vertex3, vertex4)] = score
            image_resps.append(image_responses[0])

            if config.debug:
                polygon = np.array([point1, point2, point3, point4], np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                cv2.polylines(image, [polygon], True, (0, 255, 0), 2)

                caption = "{}: {:.3f}".format(label, score)
                left_x = min(point1[0], min(point2[0], min(point3[0], point4[0])))
                top_y = min(point1[1], min(point2[1], min(point3[1], point4[1])))
                cv2.putText(image, caption, (int(left_x), int(top_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                debug_image_filename = f'test_{time.time()}.jpg'
                cv2.imwrite(debug_image_filename, image)
                print('Wrote debug image output to: "' + debug_image_filename + '"')
        counter += 1
    robot.logger.info(f"Found {len(box2conf)} bounding boxes")
    return box2conf, image_resps


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
            time.sleep(1)
        else:
            # Stand if not docked
            robot.logger.info("Commanding robot to stand...")
            blocking_stand(robot_command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")
            time.sleep(1)

        # # Unstow arm
        # if robot_state_client.get_robot_state().manipulator_state.stow_state == 1:
        #     unstow_command = RobotCommandBuilder.arm_ready_command()
        #     unstow_command_id = robot_command_client.robot_command(unstow_command)
        #     robot.logger.info("Unstow command issued.")
        #     block_until_arm_arrives(robot_command_client, unstow_command_id, 3.0)
        #     time.sleep(3)

        move_gripper(robot, robot_state_client, robot_command_client, COORD2GPOSE[6, 1], 1.0, 25)

        # Stow arm
        if robot_state_client.get_robot_state().manipulator_state.stow_state == 2:
            stow_command = RobotCommandBuilder.arm_stow_command()
            stow_command_id = robot_command_client.robot_command(stow_command)
            robot.logger.info("Stow command issued")
            block_until_arm_arrives(robot_command_client, stow_command_id, 3.0)
            time.sleep(1)

        if config.dock_after_use:
            # Dock robot after mission complete
            blocking_dock_robot(robot, config.dock_id)
            robot.logger.info("Robot docked")

        if config.poweroff_after_use:
            # Power off
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed"
            robot.logger.info("Robot safely powered off")


def move_gripper(robot, robot_state_client, robot_command_client, gripper_pose, open_fraction, hold_time, arm_command=None, collect=False):
    """
    Move arm to a spot in front of robot, then open gripper
    """
    if not arm_command:  # else use previous arm_command, only change gripper open
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
        # print(f"odom_T_hand: {odom_T_hand}\n{type(odom_T_hand)}")

        duration_seconds = 2
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z,
            odom_T_hand.rot.w, odom_T_hand.rot.x, odom_T_hand.rot.y, odom_T_hand.rot.z,
            ODOM_FRAME_NAME, duration_seconds)

    # Make open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(open_fraction)

    # Combine arm and gripper commands into 1 RobotCommand
    robot_command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

    # Send request and hold pose for 'hold_time'
    cmd_id = robot_command_client.robot_command(robot_command)
    robot.logger.info("Move arm to pose.")
    block_until_arm_arrives_with_prints(robot, robot_command_client, cmd_id)
    # block_until_arm_arrives(robot_command_client, cmd_id)
    if collect:
        while True:
            continue
    else:
        time.sleep(hold_time)
    robot.logger.info("Arm move done.")

    return arm_command


def block_until_arm_arrives_with_prints(robot, command_client, cmd_id):
    """
    Block until arm arrives at goal and print remaining distance of position and rotation.
    Note: a version w/o prints of this function is available as a helper in robot_command
    """
    while True:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        pose_dist_to_goal = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_pos_distance_to_goal
        rot_dist_to_goal = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_rot_distance_to_goal
        robot.logger.info(
            "Distance to go: " +
            f"{pose_dist_to_goal:.2f} meters" +
            f"{rot_dist_to_goal:.2f}"
        )

        if (pose_dist_to_goal < 0.01 and rot_dist_to_goal < 0.01) or feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
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


def find_center_px(vertices):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for vert in vertices:
        x, y = vert[0], vert[1]
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    x = math.fabs(max_x - min_x) / 2.0 + min_x
    y = math.fabs(max_y - min_y) / 2.0 + min_y
    return x, y


def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--username", type=str, default="user", help="Username of Spot")
    parser.add_argument("--password", type=str, default="97qp5bwpwf2c", help="Password of Spot")  # dungnydsc8su
    parser.add_argument("--dock_id", required=True, type=int, help="Docking station ID to dock at")
    parser.add_argument("--move", required=True, type=str, help="Move base or arm")
    parser.add_argument('-d', '--debug', action='store_true', help='Disable writing debug images.')
    parser.add_argument("--time_per_move", type=int, default=25, help="Seconds each move in grid should take")
    parser.add_argument('--dock_after_use', action="store_true", help='Include to dock Spot after operation')
    parser.add_argument('--poweroff_after_use', action="store_true", help='Include to power off Spot after operation')
    config = parser.parse_args(argv)
    bosdyn.client.util.setup_logging(config.verbose)

    try:
        if config.move == "base":
            move_base(config)
        elif config.move == "arm":
            move_arm(config)
        elif config.move == "test":
            test(config)
        else:
            nav_grasp(config)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.error(f"Robot control threw an exception: {exc}")
        return False


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
