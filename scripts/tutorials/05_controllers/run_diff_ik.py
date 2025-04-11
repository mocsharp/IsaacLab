# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import logging
import time
import collections
import threading
from typing import Any, Dict, Optional
from isaaclab.app import AppLauncher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#create a logger to file for logging received InputCommand IDs with timestamp
# Create a separate file logger just for tracking InputCommand message IDs
input_command_logger = logging.getLogger('input_command_tracker')
input_command_logger.setLevel(logging.INFO)
input_command_logger.addHandler(
    logging.FileHandler(f'input_command_ids_{time.strftime("%Y%m%d_%H%M%S")}.log')
)

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument("--width", type=int, default=640, help="Width of the camera.")
parser.add_argument("--height", type=int, default=480, help="Height of the camera.")
parser.add_argument("--domain_id", type=int, default=1, help="Domain ID of the camera.")
parser.add_argument(
    "--qos_provider_path",
    type=str,
    default="./dds/qos_profiles.xml",
    help="Path to the QoS provider.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

import rti.connextdds as dds
from dds.publisher import Publisher
from dds.subscriber import SubscriberWithCallback
from dds.schemas.camera_info import CameraInfo
from dds.schemas.InputCommand import InputCommand, INPUT_COMMAND_TOPIC, HIDDeviceType
import rti.idl as idl  # Import idl module for proper type handling
##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG  # isort:skip

# Message tracking variables
total_input_commands_received = 0
total_camera_info_messages_sent = 0
total_messages_ignored =0
total_failed_writes = 0
last_stats_time = time.time()
stats_interval = 1.0  # Print stats every 5 seconds

# Define the update rate
hz = 60

video_processing_thread = None
video_queue = collections.deque(maxlen=60)
class ControllerData:
    def __init__(self):
        """Stores controller data for robot and joint selection and control.
        
        Maintains state information for robot selection, joint selection,
        and incremental joint position changes based on HID input events.
        Also tracks timing information for latency measurement.
        """
        # Joint control
        self._default_joint_pos: torch.Tensor = torch.zeros(6)
        self.joint_positions: torch.Tensor = torch.zeros(6)

        # Robot selection
        self.robot_index = 0
        # Map joint names to indices
        self.joint_name_to_index = {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow": 2,
            "wrist_1": 3,
            "wrist_2": 4,
            "wrist_3": 5
        }
        
        self.joystick_values = {
            "shoulder_pan": 0.0,  # Left Joystick X
            "shoulder_lift": 0.0,  # Left Joystick Y 
            "elbow": 0.0,         # Right Joystick X
            "wrist_1": 0.0,       # Right Joystick Y
            "wrist_2": 0.0,       # ZR/ZL
            "wrist_3": 0.0        # Left/Right buttons
        }
        
        # store the latest input command received for performance measurement
        # ensure thread safe access to the latest message
        self.latest_input_command = None
        self.unique_message_ids = set()
        self.last_message_id = 0
        self.frame_id = 0
        self.message_loss_count = 0
        self._message_lock = threading.Lock()
        
    def set_incoming_message(self, message: InputCommand):
        #ensure thread safe access to the latest message
        with self._message_lock:
            self.latest_input_command = message
        
    def get_incoming_message(self):
        #ensure thread safe access to the latest message
        with self._message_lock:
            if self.latest_input_command is not None:
                input_command = self.latest_input_command
                self.latest_input_command = None
                return input_command
            return None
    
    def message_received(self, message_id: int):
        if self.last_message_id == 0:
            logger.info(f"First message received with ID: {message_id}")
        if self.last_message_id + 1 != message_id:
            logger.warning(f"Message loss detected: expected {self.last_message_id + 1}, got {message_id}")
            self.message_loss_count += 1
        self.last_message_id = message_id


    def set_robot_index(self, robot_index: int):
        """Set the robot index."""
        self.robot_index = robot_index
        
    def set_default_joint_positions(self, joint_pos: torch.Tensor):
        """Reset joint positions to default values.
        
        Args:
            joint_pos: Default joint positions tensor
        """
            
        print(f"===================set_default_joint_positions: {type(joint_pos)}")
        self._default_joint_pos = joint_pos.clone()
        self.joint_positions = joint_pos.clone()



    def set_joystick_value(self, joint_name: str, value: float):
        """Set the current value for a joystick axis.
        
        Args:
            joint_name: Name of the joint controlled by this joystick axis
            value: Normalized joystick value between -1.0 and 1.0
        """
        self.joystick_values[joint_name] = value
        
    def update_from_joysticks(self, dt: float):
        """Update joint positions based on current joystick state.
        
        Args:
            dt: Time step for the update
        """

        # Scale factor to control movement speed
        speed_factor = 0.5
        
        # Apply continuous movement based on joystick positions
        for joint_name, value in self.joystick_values.items():
            if abs(value) > 0.05:  # Add deadzone
                joint_index = self.joint_name_to_index[joint_name]
                self.joint_positions[joint_index] += value * speed_factor * dt

    def reset_joint_positions(self):
        """Reset all joint positions to 0 degrees."""
        self.joint_positions = self._default_joint_pos.clone()

class RobotCameraPublisher(Publisher):
    def __init__(self, topic: str, domain_id: int):
        """Initialize the RobotCameraPublisher.
        
        Args:
            topic: The topic to publish on
            domain_id: The DDS domain ID
        """
        super().__init__(
            topic,
            CameraInfo,
            1 / hz,
            domain_id,
            args_cli.qos_provider_path,
            "TelesurgeryLibrary::TelesurgeryApplication",
            "TelesurgeryLibrary::CameraInfo",
        )

    def produce(self, controller_data: ControllerData, camera_data: Any, input_command: InputCommand):
        """Produce a CameraInfo message with timestamp information.
        
        Args:
            controller_data: The controller data with HID timestamp information
            camera_data: The camera data to publish
            input_command: The input command to publish
            
        Returns:
            CameraInfo: The camera info message
        """
        # Create a new CameraInfo instance
        output = CameraInfo()
        if input_command is not None:
            output.message_id = input_command.message_id
            output.hid_capture_timestamp = input_command.hid_capture_timestamp
            output.hid_publish_timestamp = input_command.hid_publish_timestamp
            output.hid_receive_timestamp = input_command.hid_receive_timestamp
            output.camera_update_time = input_command.camera_update_time
            output.frame_num = controller_data.frame_id
            controller_data.frame_id += 1
        else:
            output.message_id = idl.uint64(0)
        
        output.frame_num = controller_data.frame_id
        controller_data.frame_id += 1

        # Set camera parameters
        output.robot_index = controller_data.robot_index
        output.focal_len = 12.0
        output.height = args_cli.height
        output.width = args_cli.width
        
        # Convert PyTorch tensor to numpy array and then to bytes
        if isinstance(camera_data, torch.Tensor):
            camera_data = camera_data.cpu().numpy()
        
        # output.data.extend(camera_data.tobytes())
        output.data = camera_data.tobytes()
        
        # Include joint information
        output.joint_names.extend(list(controller_data.joint_name_to_index.keys()))    
        output.joint_positions.extend(controller_data.joint_positions.tolist())
        
        # Set timestamp information
        output.camera_publish_timestamp = int(time.time_ns())
        
        # Update message tracking
        global total_camera_info_messages_sent
        total_camera_info_messages_sent += 1
        
        # Check if it's time to print stats
        global last_stats_time
        current_time = time.time()
        if current_time - last_stats_time >= stats_interval:
            # Calculate FPS based on elapsed time
            logger.info("======== Python Application Message Statistics ========")
            logger.info(f"Total InputCommand messages received: {total_input_commands_received}")
            logger.info(f"Total unique message IDs received: {len(controller_data.unique_message_ids)}")
            logger.info(f"Total messages ignored: {total_messages_ignored}")
            logger.info(f"Total failed writes: {total_failed_writes}")
            logger.info(f"Message loss count: {controller_data.message_loss_count}")
            if total_input_commands_received > 0:
                logger.info(f"Message loss rate: {controller_data.message_loss_count / total_input_commands_received * 100:.2f}%")
            last_stats_time = current_time
        
        return output

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    # UR10 robot configuration
    robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Camera configuration for UR10
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/camera",
        update_period=0.1,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),  # Position relative to UR10's end-effector
            rot=(0.5, -0.5, 0.5, -0.5),  # Quaternion rotation (w, x, y, z)
            convention="ros",
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, controller_data: ControllerData, robot_camera_publisher: RobotCameraPublisher):
    """Runs the simulation loop with joint-level robot control.
    
    Args:
        sim: The simulation context
        scene: The interactive scene
        controller_data: The controller data for HID-based joint control
    """
    # Extract scene entities
    robot = scene["robot"]
    camera = scene["camera"]

    # Specify UR10 robot-specific parameters
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=[".*"], body_names=["ee_link"]
    )
    
    # Define joint limits between -180 and +180 degrees (-π to +π radians)
    # Special case for shoulder_lift: -240 to 65 degrees (-4.18879 to 1.13446 radians)
    joint_limits_lower = torch.tensor([-3.14159, -4.18879, -3.14159, -3.14159, -3.14159, -3.14159], device=sim.device)
    joint_limits_upper = torch.tensor([3.14159, 1.13446, 3.14159, 3.14159, 3.14159, 3.14159], device=sim.device)
    
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)

    # Create markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
    )

    # Initialize robot joints
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # Initialize the joint positions in the controller data
    controller_data.set_default_joint_positions(joint_pos[controller_data.robot_index])

    # Write the initial joint state to the simulation
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # EPS calculation variables
    loop_count = 0
    last_eps_report_time = time.perf_counter()
    eps_report_interval = 2.0  # Report every 2 seconds
    # Variables for overall EPS statistics
    total_eps_sum = 0.0
    interval_count = 0
    min_eps = float('inf')
    max_eps = 0.0

    # Simulation loop
    while simulation_app.is_running():
        # Increment loop counter
        loop_count += 1

        event = controller_data.get_incoming_message()

        # # Update joint positions based on joystick state
        controller_data.update_from_joysticks(sim_dt)
        
        # Get the current joint positions
        joint_pos = robot.data.joint_pos.clone()
        
        # Apply controller data joint positions using vectorized clamping
        # Clamp the controller's joint positions based on the defined limits
        clamped_joint_positions = torch.clamp(
            controller_data.joint_positions, min=joint_limits_lower, max=joint_limits_upper
        )
        # Update the controller data with the clamped positions
        controller_data.joint_positions = clamped_joint_positions
        
        # Apply the clamped joint positions to the specific robot being controlled
        joint_pos[controller_data.robot_index] = clamped_joint_positions
                    
        # Apply joint positions directly
        robot.set_joint_position_target(joint_pos)
        
        # Write to simulation
        scene.write_data_to_sim()

        # Step simulation
        sim.step()
        
        # Update scene
        scene.update(sim_dt)
        
        # Get end-effector pose for visualization
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        
        # Update and process camera data
        camera.update(sim_dt)
        
        # Capture timestamp before stepping simulation
        if event is not None:
            event.camera_update_time = int(time.time_ns())
        
        # Access camera data from the selected robot
        rgb_data = camera.data.output.get("rgb")
        
        if rgb_data is not None:
            video_queue.append((rgb_data[controller_data.robot_index], event))
            
        # Calculate and report EPS periodically
        current_time = time.perf_counter()
        elapsed_time = current_time - last_eps_report_time
        if elapsed_time >= eps_report_interval:
            eps = loop_count / elapsed_time
            # Update overall statistics
            total_eps_sum += eps
            interval_count += 1
            min_eps = min(min_eps, eps)
            max_eps = max(max_eps, eps)
            avg_eps = total_eps_sum / interval_count

            logger.info(f"Simulation loop EPS: Current={eps:.2f}, Avg={avg_eps:.2f}, Min={min_eps:.2f}, Max={max_eps:.2f}")
            # Reset for next interval
            loop_count = 0
            last_eps_report_time = current_time

def hid_callback(topic: str, controller_data: ControllerData, data: InputCommand):
    """Process HID events for joint-level robot control.
    
    Args:
        topic: The topic on which the message was received
        controller_data: The controller data instance to update
        data: The input command data received
    """
    # logger.info(f"Received HID event: {data}")
    time_now = time.time_ns()
    
    # Update message tracking
    global total_input_commands_received
    total_input_commands_received += 1
    controller_data.message_received(data.message_id)
    message_handled = False
    
    match data.device_type:
        case HIDDeviceType.JOYSTICK:
            match data.event_type:
                case 1:  # buttons
                    match data.number:
                        case 0 | 1 | 2 | 3:  # A/B/X/Y - Cycle through robots
                            if data.value != 0:
                                robot_index = (controller_data.robot_index + 1) % args_cli.num_envs
                                controller_data.set_robot_index(robot_index)
                                message_handled = True
                                # logger.info(f"Selected robot: {controller_data.robot_index}")
                        case 5:  # left - controls wrist_3 (negative direction)
                            if data.value != 0:  # Button pressed
                                controller_data.set_joystick_value("wrist_3", -1.0)
                                message_handled = True
                                # logger.info(f"Set wrist_3 joystick: -1.0 (left button)")
                            else:  # Button released
                                controller_data.set_joystick_value("wrist_3", 0.0)
                                message_handled = True
                                # logger.info(f"Reset wrist_3 joystick (left button released)")
                        case 6:  # right - controls wrist_3 (positive direction)
                            if data.value != 0:  # Button pressed
                                controller_data.set_joystick_value("wrist_3", 1.0)
                                message_handled = True
                                # logger.info(f"Set wrist_3 joystick: 1.0 (right button)")
                            else:  # Button released
                                controller_data.set_joystick_value("wrist_3", 0.0)
                                message_handled = True
                                # logger.info(f"Reset wrist_3 joystick (right button released)")
                        case 8: # reset all positions to 0 degrees
                            controller_data.reset_joint_positions()
                            message_handled = True
                            # logger.info("Reset all joint positions to 0 degrees")

                        # NV button (4), Circle (9), L Triangle (7), R Triangle (8) have no function
                case 2:  # Joysticks and D-Pad
                    # Normalize value to [-1.0, 1.0]
                    normalized_value = data.value / 32767.0
                    match data.number:
                        case 0: # Left Joystick X - controls shoulder_pan
                            controller_data.set_joystick_value("shoulder_pan", normalized_value)
                            message_handled = True
                            # logger.info(f"Set shoulder_pan joystick: {normalized_value}")
                        case 1: # Left Joystick Y - controls shoulder_lift
                            controller_data.set_joystick_value("shoulder_lift", normalized_value)
                            message_handled = True
                            # logger.info(f"Set shoulder_lift joystick: {normalized_value}")
                        case 2: # Right Joystick X - controls elbow
                            controller_data.set_joystick_value("elbow", normalized_value)
                            message_handled = True
                            # logger.info(f"Set elbow joystick: {normalized_value}")
                        case 5: # Right Joystick Y - controls wrist_1
                            controller_data.set_joystick_value("wrist_1", normalized_value)
                            message_handled = True
                            # logger.info(f"Set wrist_1 joystick: {normalized_value}")
                        case 3: # ZR - controls wrist_2 (positive direction)
                            controller_data.set_joystick_value("wrist_2", abs(normalized_value))
                            message_handled = True
                            # logger.info(f"Set wrist_2 joystick: {abs(normalized_value)} (ZR)")
                        case 4: # ZL - controls wrist_2 (negative direction)
                            controller_data.set_joystick_value("wrist_2", -abs(normalized_value))
                            message_handled = True
                            # logger.info(f"Set wrist_2 joystick: {-abs(normalized_value)} (ZL)")
                        case 6 | 7: # ZR/ZL - stop wrist_2 movement
                            if normalized_value == -1.0:
                                controller_data.set_joystick_value("wrist_2", 0.0)
                                message_handled = True
                                # logger.info(f"Reset wrist_2 joystick")
    if not message_handled:
        global total_messages_ignored
        total_messages_ignored += 1
    else:
        data.hid_receive_timestamp = time_now
        controller_data.set_incoming_message(data)

    input_command_logger.info(f"{time.time()}: {data.message_id}\t{'1' if message_handled else '0'}")

def process_video_queue(controller_data: ControllerData, robot_camera_publisher: RobotCameraPublisher):
    """Process the video queue."""
    # Track write calls per second and average size of the queue
    write_count = 0
    last_print_time = time.time()
    queue_size = []
    while True:
        if len(video_queue) > 0:
            queue_size.append(len(video_queue))
            camera_data, event = video_queue.popleft()
            try:
                robot_camera_publisher.write(controller_data, camera_data, event)
            except Exception as e:
                logger.error(f"Error writing to robot_camera_publisher: {e}")
            write_count += 1
        else:
            time.sleep(1/120)
        if len(queue_size) > 0 and time.time() - last_print_time >= 1.0:
            logger.info(f"Camera publisher write calls per second: {write_count}")
            logger.info(f"Average queue size: {sum(queue_size) / len(queue_size)}")
            queue_size = []
            write_count = 0
            last_print_time = time.time()

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    logger.info("[INFO]: Setup complete...")
    controller_data = ControllerData()
    # Configure HID subscriber
    input_command_subscriber = SubscriberWithCallback(
        hid_callback,
        args_cli.domain_id,
        INPUT_COMMAND_TOPIC,
        InputCommand,
        1 / hz,
        args_cli.qos_provider_path,
        "TelesurgeryLibrary::TelesurgeryApplication",
        "TelesurgeryLibrary::InputCommand",
        controller_data
    )
    input_command_subscriber.start()

    robot_camera_publisher = RobotCameraPublisher(
        topic="robot_camera",
        domain_id=args_cli.domain_id,
    )
    video_processing_thread = threading.Thread(target=process_video_queue, args=(controller_data, robot_camera_publisher))
    video_processing_thread.start()
    # Run the simulator
    run_simulator(sim, scene, controller_data, robot_camera_publisher)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
