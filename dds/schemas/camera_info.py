
# WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

# This file was generated from camera_info.idl
# using RTI Code Generator (rtiddsgen) version 4.3.0.
# The rtiddsgen tool is part of the RTI Connext DDS distribution.
# For more information, type 'rtiddsgen -help' at a command shell
# or consult the Code Generator User's Manual.

from dataclasses import field
from typing import Union, Sequence, Optional
import rti.idl as idl
from enum import IntEnum
import sys
import os


@idl.struct(
    member_annotations = {
        'joint_names': [idl.bound(6), idl.element_annotations([idl.unbounded])],
        'joint_positions': [idl.bound(6)],
    }
)
class CameraInfo:
    focal_len: float = 0.0
    robot_index: int = 0
    stream_id: int = 0
    frame_num: int = 0
    width: int = 0
    height: int = 0
    data: Sequence[idl.uint8] = field(default_factory = idl.array_factory(idl.uint8))
    joint_names: Sequence[str] = field(default_factory = list)
    joint_positions: Sequence[float] = field(default_factory = idl.array_factory(float))
    capture_timestamp: int = 0
    hid_publish_timestamp: int = 0
    receive_timestamp: int = 0
    camera_update_time: int = 0
    camera_publish_timestamp: int = 0
    message_id: idl.uint64 = 0
