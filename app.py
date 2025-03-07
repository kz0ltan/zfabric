#!/usr/bin/env python3
"""zFabric API server"""

import argparse

from server import FabricAPIServer


def parse_arguments():
    """ArgParse argument parsing"""
    parser = argparse.ArgumentParser(description="zFabric server application")

    parser.add_argument(
        "-l", "--listen", type=str, default="localhost", help="Hostname/IP to listen on"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=13337, help="Port to listen on"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Werkzeug debug mode"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./config.json",
        help="Path to config JSON file",
    )

    return parser.parse_args()


def start():
    """Meant to be used by Gunicorn"""
    return FabricAPIServer().app


if __name__ == "__main__":
    cli_args = parse_arguments()
    FabricAPIServer("zFabric", cli_args.config).app.run(
        host=cli_args.listen, port=cli_args.port, debug=cli_args.debug
    )
