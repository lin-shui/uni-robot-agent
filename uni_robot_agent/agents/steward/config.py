"""steward (DeepSeek Harness) settings; dsh owns model transport and the loop."""
from __future__ import annotations

import argparse
import os
from urllib.parse import urlsplit


class StewardConfig:
    """CLI surface for the dsh agent backend, resolved before any runtime starts."""

    model_protocol = "responses"

    @classmethod
    def add_cli_arguments(cls, parser):
        group = parser.add_argument_group("steward (DeepSeek Harness) agent")
        group.add_argument(
            "--steward-dsh-home", default=argparse.SUPPRESS,
            help="Harness home for the dsh runtime (default: DSH_HOME)",
        )
        group.add_argument(
            "--steward-profile", default=argparse.SUPPRESS,
            help="dsh profile hosting the SDK JSON-RPC server (default: sdk)",
        )
        group.add_argument(
            "--steward-patch", action="append", default=argparse.SUPPRESS, metavar="PATH",
            help="Extra cordis patch applied after the generated agent patch",
        )
        group.add_argument(
            "--steward-provider", default=argparse.SUPPRESS,
            help="dsh model route name (default: steward when a base URL is given)",
        )
        group.add_argument(
            "--steward-base-url", default=argparse.SUPPRESS,
            help="OpenAI-compatible endpoint dsh calls; --provider-url is also accepted",
        )
        group.add_argument(
            "--steward-context-window", type=int, default=argparse.SUPPRESS,
            help="Declared context window of the model route (default: 262144)",
        )
        group.add_argument(
            "--steward-max-tokens", type=int, default=argparse.SUPPRESS,
            help="Declared output-token capability of the model route (default: 32768)",
        )
        group.add_argument(
            "--steward-native-tools", action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help="Keep dsh shell/file/skill tools beside the robot tools (default: on)",
        )

    @classmethod
    def resolve_defaults(cls, args):
        base_url = getattr(args, "steward_base_url", None) or args.provider_url
        args.steward_base_url = base_url
        args.steward_provider = getattr(args, "steward_provider", None) or (
            "steward" if base_url else None
        )
        args.steward_dsh_home = (
            getattr(args, "steward_dsh_home", None) or os.environ.get("DSH_HOME")
        )
        args.steward_profile = getattr(args, "steward_profile", None) or "sdk"
        args.steward_patch = tuple(getattr(args, "steward_patch", None) or ())
        args.steward_context_window = getattr(args, "steward_context_window", 262144)
        args.steward_max_tokens = getattr(args, "steward_max_tokens", 32768)
        args.steward_native_tools = getattr(args, "steward_native_tools", True)
        if args.reasoning_effort is None:
            args.reasoning_effort = "none"
        # dsh owns model transport; the shared provider gateway must stay closed.
        args.provider_url = None

    @classmethod
    def validate_config(cls, args, explicit):
        if args.context_policy != "identity":
            raise ValueError("steward currently supports only the identity context policy")
        if not args.steward_dsh_home:
            raise ValueError("steward requires --steward-dsh-home or DSH_HOME")
        if args.steward_base_url:
            if urlsplit(args.steward_base_url).scheme not in {"http", "https"}:
                raise ValueError("steward model base URL must use HTTP or HTTPS")
        if not args.steward_provider:
            raise ValueError("steward requires --steward-provider or --steward-base-url")
        if args.steward_context_window <= 0:
            raise ValueError("--steward-context-window must be positive")
        if args.steward_max_tokens <= 0:
            raise ValueError("--steward-max-tokens must be positive")

    @classmethod
    def config_options(cls, args):
        return {
            "dsh_home": args.steward_dsh_home,
            "profile": args.steward_profile,
            "patches": args.steward_patch,
            "provider": args.steward_provider,
            "base_url": args.steward_base_url,
            "api_key": args._api_key,
            "context_window": args.steward_context_window,
            "max_tokens": args.steward_max_tokens,
            "native_tools": args.steward_native_tools,
            "cwd": getattr(args, "output_dir", None),
        }
