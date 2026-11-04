"""Cobot Magic twin integration: shared gripper tools over a WSim episode."""
from __future__ import annotations

from argparse import BooleanOptionalAction

from uni_robot_agent.integrations.gripper import GripperOptions


class CobotMagicOptions(GripperOptions):
    pass


def add_cli_args(parser):
    from uni_robot_agent.integrations.gripper import add_gripper_cli_args

    add_gripper_cli_args(parser)


def validate_cli(args):
    from uni_robot_agent.integrations.gripper import validate_gripper_cli

    if args.vla_review is None:
        args.vla_review = args.action_profile == "hybrid"
    validate_gripper_cli(args)


def create_session(context):
    from uni_robot_agent.integrations.gripper import create_session as build

    return build(context)


def get_integration_spec():
    from uni_robot_agent.integrations.gripper import build_gripper_spec

    return build_gripper_spec(name="cobot_magic", create_session=create_session,
                              add_cli_args=add_cli_args, validate_cli=validate_cli,
                              config_model=CobotMagicOptions)
