
# WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

# This file was generated from InputCommand.idl
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


INPUT_COMMAND_TOPIC = "InputCommand"

@idl.enum
class HIDDeviceType(IntEnum):
    JOYSTICK = 0
    KEYBOARD = 1
    MOUSE = 2

@idl.struct(
    member_annotations = {
        'device_name': [idl.key, idl.bound(255)],
    }
)
class InputCommand:
    device_name: str = ""
    device_type: HIDDeviceType = HIDDeviceType.JOYSTICK
    event_type: idl.uint8 = 0
    number: idl.uint8 = 0
    value: idl.int16 = 0
    capture_timestamp: int = 0
    publish_timestamp: int = 0
    message_id: idl.uint64 = 0
