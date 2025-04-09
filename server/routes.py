import datetime
import itertools
import json
import os
from typing import Optional, Union, Dict, List
from pathlib import Path

from flask import request, jsonify, Response
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages import message_to_dict

from .decorators import auth_required
from .helpers import load_file


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

    @server.app.route("/contexts", methods=["GET"])
    @auth_required(server)
    def list_contexts():
        contexts = [
            cfile
            for cpath in server.config["context_paths"]
            if os.path.isdir(cpath)
            for cfile in os.listdir(cpath)
            if os.path.isfile(os.path.join(cpath, cfile))
        ]

        return jsonify({"response": list(contexts)})

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

        server.app.logger.debug(
            "Milling request received.\n\tArguments: " + str(request.args)
        )

        profile_name = request.args.get("profile", None)
        model = request.args.get("model", None)
        options = request.args.get("options", None)
        stream = request.args.get(
            "stream",
            default=False,
            type=(lambda s: s.lower() in ("True", "true", "1")),
        )
        keep_alive = request.args.get("keep_alive", default=None)
        session = request.args.get(
            "session",
            default=str(pattern)
            + datetime.datetime.today().strftime("-%Y-%m-%d_%H-%M-%S-%f"),
            # + str(generate_random_number(5)),
        )
        # skip saving session
        if session == "skip":
            session = None

        variables: Dict[str, Union[str, int]] = request.args.get(
            "variables", default={}
        )

        if options:
            options = json.loads(options)

        contexts: List[str] = (
            request.args.get("contexts", "").split(",")
            if request.args.get("contexts", "")
            else []
        )

        data = request.get_json()
        if len(data) == 0:
            return jsonify({"error": "Missing input data"}), 400

        input_attachment = None
        if "attachment" in data:
            input_attachment = data["attachment"]
            del data["attachment"]

        messages = []
        new_messages = []
        user_prompt = ""
        timestamp = datetime.datetime.now().timestamp()

        if session in server.session_manager.get_session_names():
            messages += server.session_manager.get_session(session).messages

        if pattern is not None:
            for ppath in server.config["pattern_paths"]:
                pattern_path = Path(ppath) / pattern
                if pattern_path.exists():
                    break
                if ppath == server.config["pattern_paths"][-1]:
                    return jsonify({"error": f"Pattern not found: {pattern}"}), 400

            if len(contexts) > 0:
                for context in contexts:
                    context_split = context.split(":", 1)
                    tag = None
                    if len(context_split) > 1:
                        tag, context = context_split
                        if len(tag) == 0:
                            tag = None
                    for cpath in server.config["context_paths"]:
                        context_path = Path(cpath) / (context + ".md")
                        if context_path.exists():
                            data[tag if tag else "context_" + context] = load_file(
                                context_path
                            )
                            break
                        if cpath == server.config["context_paths"][-1]:
                            return jsonify(
                                {"error": f"Context not found: {context}"}
                            ), 400

            # load system prompt; system prompt only works in empty (new) sessions
            if len(messages) == 0:
                system_prompt = load_file(pattern_path / "system.md", "")
                system_prompt = server.variable_handler.resolve(
                    system_prompt, variables, data
                )

                if len(system_prompt):
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

            # load user prompt
            user_prompt = load_file(pattern_path / "user.md", "")

        user_prompt = server.variable_handler.resolve(
            user_prompt, variables, data, coalesce_data=True
        )

        content = []
        if len(user_prompt):
            content.append(
                {
                    "type": "text",
                    "text": user_prompt,
                }
            )
        if input_attachment:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{input_attachment}"},
                }
            )
        if len(content):
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

        return server.generator.generate(
            profile_name,
            messages,
            options,
            stream,
            keep_alive=keep_alive,
            model=model,
            session=session,
        )
