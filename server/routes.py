import datetime
import itertools
import json
import os
from typing import Optional, Union, Dict
from pathlib import Path

from flask import request, jsonify
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages import message_to_dict

from .decorators import auth_required
from .helpers import generate_random_number, load_file


def register_routes(server):
    """Register routes for the server application"""

    @server.app.route("/tokens", methods=["POST"])
    @auth_required(server)
    def count_tokens():
        text = request.get_json()["input"]
        return jsonify({"response": str(server.generator.count_tokens(text))})

    @server.app.route("/profiles", methods=["GET"])
    @auth_required(server)
    def list_profiles():
        profiles = list(server.config["profiles"].keys())
        if "default_profile" in profiles:
            profiles.remove("default_profile")
            default_profile = server.config["profiles"]["default_profile"]
            profiles.insert(0, "Default: " + str(default_profile))
        return jsonify({"response": profiles})

    @server.app.route("/patterns", methods=["GET"])
    @auth_required(server)
    def list_patterns():
        patterns = itertools.chain.from_iterable(
            [
                os.listdir(ppath)
                for ppath in server.config["pattern_paths"]
                if os.path.isdir(ppath)
            ]
        )
        return jsonify({"response": list(patterns)})

    @server.app.route("/sessions", methods=["GET"])
    @auth_required(server)
    def list_session():
        return jsonify({"response": server.session_manager.get_session_names()})

    @server.app.route("/session/<session>", methods=["GET", "DELETE"])
    @auth_required(server)
    def get_session(session: str):
        if request.method == "GET":
            return jsonify(
                {
                    "response": [
                        message_to_dict(msg)
                        for msg in server.session_manager.get_session(session).messages
                    ]
                }
            )

        if request.method == "DELETE":
            server.session_manager.delete_session(session)
            return jsonify({"response": ""})

    @server.app.route("/patterns/<pattern>", methods=["POST"])
    @server.app.route("/session", methods=["POST"])
    @auth_required(server)
    def milling(pattern: Optional[str] = None):
        """Combine fabric pattern with input from user and send to OpenAI's GPT-4 model.

        Returns:
            JSON: A JSON response containing the generated response or an error message.

        Raises:
            Exception: If there is an error during the API call.
        """

        profile_name = request.args.get("profile", None)
        model = request.args.get("model", None)
        options = request.args.get("options", None)
        stream = request.args.get(
            "stream",
            default=False,
            type=(lambda s: s.lower() in ("True", "true", "1")),
        )
        session = request.args.get(
            "session",
            default=datetime.datetime.today().strftime("%Y-%m-%d-")
            + str(generate_random_number(5)),
        )
        # skip saving session
        if session == "skip":
            session = None

        variables: Dict[str, Union[str, int]] = request.args.get(
            "variables", default={}
        )

        if options:
            options = json.loads(options)

        data = request.get_json()
        if "input" not in data:
            return jsonify({"error": "Missing input parameter"}), 400

        input_data = data["input"]
        input_attachment = None
        if "attachment" in data:
            input_attachment = data["attachment"]

        messages = []
        new_messages = []
        user_prompt = ""
        timestamp = datetime.datetime.now().timestamp()

        if session in server.session_manager.get_session_names():
            messages += server.session_manager.get_session(session).messages
        elif pattern is not None:
            # session exists, patterns are used
            for ppath in server.config["pattern_paths"]:
                pattern_path = Path(ppath) / pattern
                if pattern_path.exists():
                    break
                if ppath == server.config["pattern_paths"][-1]:
                    return jsonify({"error": "Pattern not found"}), 400

            system_prompt = load_file(pattern_path / "system.md", "")
            user_prompt = load_file(pattern_path / "user.md", "")

            system_prompt = server.variable_handler.resolve(system_prompt, variables)
            user_prompt = server.variable_handler.resolve(user_prompt, variables)

            # Build the API call
            # https://python.langchain.com/api_reference/core/messages/langchain_core.messages.chat.ChatMessage.html
            system_message = ChatMessage(
                content=system_prompt,
                role="system",
                response_metadata={
                    "options": options,
                    "session": session,
                    "timestamp": timestamp,
                },
            )
            new_messages.append(system_message)

        content = [
            {
                "type": "text",
                "text": (user_prompt + "\n")
                if len(user_prompt) > 0
                else "" + input_data,
            }
        ]
        if input_attachment:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{input_attachment}"},
                }
            )
        user_message = ChatMessage(
            content=content,
            role="user",
            response_metadata={
                "options": options,
                "session": session,
                "timestamp": timestamp,
            },
        )
        new_messages.append(user_message)
        messages += new_messages
        server.session_manager.add_messages(session, new_messages, merge=False)

        try:
            return server.generator.generate(
                profile_name,
                messages,
                options,
                stream,
                model=model,
                session=session,
            )
        except Exception as ex:
            server.logger.error("Error occured: %s", ex)
            return jsonify(
                {"error": "An error occurred while processing the request."}
            ), 500
