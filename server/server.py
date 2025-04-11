"""zFabric API server"""

import json
import logging
import os
from typing import Optional
import traceback

from flask import Flask, jsonify

from server.routes import register_routes
from server.models import SessionManager, VariableHandler
from server.services import Generator
from server.helpers import load_file


class FabricAPIServer:
    """Flask API server implementing all the server-side functionality of zFabric"""

    def __init__(self, name: str = "zFabric", config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH")
        self.config = json.loads(load_file(config_path))

        self.app = Flask(name)
        self.app.logger.setLevel(self.config.get("loglevel", logging.INFO))
        register_routes(self)
        self.add_errorhandlers()

        self.variable_handler = VariableHandler(self.config)
        self.session_manager = SessionManager(self.config)
        self.generator = Generator(self.config, smanager=self.session_manager)

    def check_auth_token(self, token: str):
        """Verify authentication token"""
        user_db = self.config["users"]
        for user in user_db.keys():
            if user_db[user]["api_key"] == token:
                return user
        return None

    def add_errorhandlers(self):
        """Register Flask error handlers"""

        @self.app.errorhandler(404)
        def not_found(exception):
            return jsonify({"error": "The requested resource was not found."}), 404

        @self.app.errorhandler(Exception)
        def handle_exception(exception):
            self.app.logger.error("Error occured: %s", exception)
            stack_trace = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
            self.app.logger.debug(stack_trace)

            # stream = request.args.get(
            #    "stream", "false").lower() in ("true", "1")
            # if stream:
            #    def generate_error():
            #        yield json.dumps({"error": "An error occurred while processing the request"})
            #    return Response(generate_error(), mimetype="application/json")
            # else:
            return json.dumps({"error": str(exception)}), 500
