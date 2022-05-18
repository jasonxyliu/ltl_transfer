# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import sys
import os
import time
from collections import defaultdict
import cv2
import numpy as np
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request

IMG_SOURCES = [
    # 'frontleft_fisheye_image',
    # 'frontright_fisheye_image',
    # 'left_fisheye_image',
    # 'right_fisheye_image',
    # 'back_fisheye_image',
    'hand_color_image', 'hand_depth_in_hand_color_frame', 'hand_color_in_hand_depth_frame', 'hand_depth', 'hand_image'
]

HAND_IMG_SOURCES = [
    'hand_color_image', 'hand_depth_in_hand_color_frame', 'hand_color_in_hand_depth_frame', 'hand_depth', 'hand_image'
]


def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--username', type=str, default='user', help='Username of Spot')
    parser.add_argument('--password', type=str, default='97qp5bwpwf2c', help='Password of Spot')  # dungnydsc8su
    parser.add_argument('--image-source', default='hand_color_image', help=f'Get image from source(s). Available image sources: {str(IMG_SOURCES)}')
    parser.add_argument('--image_dpath', default='./multiobj/images', help='Path to write images to')
    parser.add_argument('--all_hand_cameras', action="store_true", help='Include to capture and save images from all hand camera')
    options = parser.parse_args(argv)

    # Make sure the folder exists.
    if not os.path.exists(options.image_dpath):
        print(f'output image folder does not exist. made one at {options.image_dpath}')
        os.makedirs(options.image_dpath, exist_ok=True)

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    robot.authenticate(username=options.username, password=options.password)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    src2counter = defaultdict(int)

    while True:
        if options.all_hand_cameras:
            # We want to capture from all (hand) cameras.

            # Capture and save images to disk
            image_responses = image_client.get_image_from_sources(HAND_IMG_SOURCES)

            for image_response in image_responses:
                # Unpack image
                img = np.frombuffer(image_response.shot.image.data, dtype=np.uint8)
                img = cv2.imdecode(img, -1)

                # Optionally rotate image to level. Only left and back cameras on Spot are upright
                img_src_name = image_response.source.name
                if image_response.source.name[0:5] == "front" or img_src_name in HAND_IMG_SOURCES[2:]:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif image_response.source.name[0:5] == "right":
                    img = cv2.rotate(img, cv2.ROTATE_180)

                # Avoid overwriting an existing image
                while True:
                    image_saved_path = os.path.join(options.image_dpath, image_response.source.name + f'_{src2counter[img_src_name]:0>4d}.jpg')
                    src2counter[img_src_name] += 1

                    if not os.path.exists(image_saved_path):
                        break

                print(image_saved_path)

                # Save image
                if img is not None:
                    cv2.imwrite(image_saved_path, img)
                    print(f'Wrote: {image_saved_path}')
        else:
            # We want to capture from one camera at a time.

            # Capture and save images to disk
            image_responses = image_client.get_image_from_sources([options.image_source])

            # Unpack image
            img = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
            img = cv2.imdecode(img, -1)

            # Optionally rotate image to level. Only left and back cameras on Spot are upright
            img_src_name = image_responses[0].source.name
            if image_responses[0].source.name[0:5] == "front" or img_src_name in HAND_IMG_SOURCES[2:]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif image_responses[0].source.name[0:5] == "right":
                img = cv2.rotate(img, cv2.ROTATE_180)

            # Avoid overwriting an existing image
            while True:
                image_saved_path = os.path.join(options.image_dpath, image_responses[0].source.name + f'_{src2counter[img_src_name]:0>4d}.jpg')
                src2counter[img_src_name] += 1

                if not os.path.exists(image_saved_path):
                    break

            # Save image
            cv2.imwrite(image_saved_path, img)
            print(f'Wrote: {image_saved_path}')

        # Wait for some time so we can drive the robot to a new position.
        time.sleep(0.7)


def main_one_camera(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--username', type=str, default='user', help='Username of Spot')
    parser.add_argument('--password', type=str, default='97qp5bwpwf2c', help='Password of Spot')  # dungnydsc8su
    parser.add_argument('--image-source', default='hand_color_image', help=f'Get image from source(s). Available image sources: {str(IMG_SOURCES)}')
    parser.add_argument('--image_dpath', default='./multiobj/images', help='Path to write images to')
    options = parser.parse_args(argv)

    # Make sure the folder exists.
    if not os.path.exists(options.image_dpath):
        print(f'output image folder does not exist. made one at {options.image_dpath}')
        os.makedirs(options.image_dpath, exist_ok=True)

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    robot.authenticate(username=options.username, password=options.password)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    counter = 0

    while True:
        # We want to capture from one camera at a time.

        # Capture and save images to disk
        image_responses = image_client.get_image_from_sources([options.image_source])

        # Unpack image
        img = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
        img = cv2.imdecode(img, -1)

        # Optionally rotate image to level. Only left and back cameras on Spot are upright
        img_src_name = image_responses[0].source.name
        if image_responses[0].source.name[0:5] == "front" or img_src_name in HAND_IMG_SOURCES[2:]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif image_responses[0].source.name[0:5] == "right":
            img = cv2.rotate(img, cv2.ROTATE_180)

        # Avoid overwriting an existing image
        while True:
            image_saved_path = os.path.join(options.folder, image_responses[0].source.name + f'_{counter:0>4d}.jpg')
            counter += 1

            if not os.path.exists(image_saved_path):
                break

        # Save image
        cv2.imwrite(image_saved_path, img)
        print(f'Wrote: {image_saved_path}')

        # Wait for some time so we can drive the robot to a new position.
        time.sleep(0.7)


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
