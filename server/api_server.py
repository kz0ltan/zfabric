#!/usr/bin/env python3
"""zFabric API server"""

import argparse
from base64 import b64encode
from functools import wraps
import json
import itertools
import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Iterable, Any

from flask import Flask, request, jsonify, Response
from flask.logging import default_handler
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.utils import merge_message_runs
from langchain_core.messages import message_to_dict
from langchain_community.chat_message_histories import SQLChatMessageHistory
import ollama
import openai
from sqlalchemy import create_engine, select, distinct, MetaData
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select
import tiktoken

from helpers import load_file, generate_random_number


class FabricAPIServer:
    """Flask API server implementing all the server-side functionality of Fabric"""

    def __init__(self, name: str = "zFabric", config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH")
        self.config = json.loads(load_file(config_path))

        self.app = Flask(name)
        self.app.logger.setLevel(logging.INFO)
        self.add_routes()
        self.add_errorhandlers()

        self.variable_handler = VariableHandler(self.config)
        self.session_manager = SessionManager(self.config)
        self.generator = AnswerGenerator(self.config, smanager=self.session_manager)

    def check_auth_token(self, token: str):
        """Verify authentication token"""
        user_db = self.config["users"]
        for user in user_db.keys():
            if user_db[user]["api_key"] == token:
                return user
        return None

    def auth_required(self, func):
        """Decorator function to check if the token is valid.

        Args:
            f: The function to be decorated

        Returns:
            The decorated function
        """

        @wraps(func)
        def decorated_function(*args, **kwargs):
            """Decorated function to handle authentication token and API endpoint.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Result of the decorated function.

            Raises:
                KeyError: If 'Authorization' header is not found in the request.
                TypeError: If 'Authorization' header value is not a string.
                ValueError: If the authentication token is invalid or expired.
            """

            # Get the authentication token from request header
            auth_token = request.headers.get("Authorization", "")

            # Remove any bearer token prefix if present
            if auth_token.lower().startswith("bearer "):
                auth_token = auth_token[7:]

            # Check if token is valid
            user = self.check_auth_token(auth_token)
            if user is None:
                return jsonify({"error": "user not found!"}), 401

            return func(*args, **kwargs)

        return decorated_function

    def add_errorhandlers(self):
        """Register Flask error handlers"""

        @self.app.errorhandler(404)
        def not_found():
            return jsonify({"error": "The requested resource was not found."}), 404

        @self.app.errorhandler(500)
        def server_error(exception):
            """Manually raise an internal server error:
            flask.abort(500)
            """
            self.app.logger.error("Error occured: %s", exception)
            return jsonify({"error": "An internal server error occurred."}), 500

    def add_routes(self):
        """Add routes to the application"""

        @self.app.route("/profiles", methods=["GET"])
        @self.auth_required
        def list_profiles():
            profiles = list(self.config["profiles"].keys())
            if "default_profile" in profiles:
                profiles.remove("default_profile")
                default_profile = self.config["profiles"]["default_profile"]
                profiles.insert(0, "Default: " + str(default_profile))
            return jsonify({"response": profiles})

        @self.app.route("/patterns", methods=["GET"])
        @self.auth_required
        def list_patterns():
            patterns = itertools.chain.from_iterable(
                [
                    os.listdir(ppath)
                    for ppath in self.config["pattern_paths"]
                    if os.path.isdir(ppath)
                ]
            )
            return jsonify({"response": list(patterns)})

        @self.app.route("/sessions", methods=["GET"])
        @self.auth_required
        def list_session():
            return jsonify({"response": self.session_manager.get_session_names()})

        @self.app.route("/session/<session>", methods=["GET"])
        @self.auth_required
        def get_session(session: str):
            return jsonify(
                {
                    "response": [
                        message_to_dict(msg)
                        for msg in self.session_manager.get_session(session).messages
                    ]
                }
            )

        @self.app.route("/patterns/<pattern>", methods=["POST"])
        @self.auth_required
        def milling(pattern: str):
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
                default=generate_random_number(10),
            )
            variables = request.args.get("variables", default={})

            if options:
                options = json.loads(options)

            data = request.get_json()
            if "input" not in data:
                return jsonify({"error": "Missing input parameter"}), 400

            input_data = data["input"]

            for ppath in self.config["pattern_paths"]:
                pattern_path = Path(ppath) / pattern
                if pattern_path.exists():
                    break
                if ppath == self.config["pattern_paths"][-1]:
                    return jsonify({"error": "Pattern not found"}), 400

            system_prompt = load_file(pattern_path / "system.md", "")
            user_prompt = load_file(pattern_path / "user.md", "")

            system_prompt = self.variable_handler.resolve(system_prompt, variables)
            user_prompt = self.variable_handler.resolve(user_prompt, variables)

            # Build the API call
            # https://python.langchain.com/api_reference/core/messages/langchain_core.messages.chat.ChatMessage.html
            system_message = ChatMessage(content=system_prompt, role="system")
            user_message = ChatMessage(
                content=user_prompt + "\n" + input_data, role="system"
            )
            messages = [system_message, user_message]

            try:
                return self.generator.generate(
                    profile_name, model, messages, options, stream, session
                )
            except Exception as e:
                self.app.logger.error("Error occured: %s", e)
                raise e
                # return jsonify({"error": "An error occurred while processing the request."}), 500


class SessionManager:
    """
    Handles sessions (single query or chat)
    * Save/update sessions using configured storage mechanism
    * Works with langchain to store / retrieve sessions
    """

    def __init__(self, config: Dict[Any, Any], session_id_field_name: str = ""):
        self.db_path = config.get("sqlite3_db_path", "/zfabric.sqlite3")
        self.table_name = config.get("sqlite3_table_name", "message_store")
        self.session_id_field_name = config.get(
            "sqlite3_session_id_field_name", "session_id"
        )
        self._db_connection = None
        self.metadata = None
        self.table = None

        self.logger = logging.getLogger("app.variables")
        self.logger.addHandler(default_handler)
        self.logger.setLevel(logging.INFO)

    @property
    def db_connection(self):
        """Lazy setup of db connection"""
        if self._db_connection:
            return self._db_connection
        if len(self.db_path) == 0:
            self.logger.warning(
                "No 'sqlite3_db_path' was found in config, using volatile, memory storage!"
            )
        else:
            self.db_path = os.path.expanduser(self.db_path)
        self._db_connection = self._setup_db_connection("sqlite://" + self.db_path)
        return self._db_connection

    def _setup_db_connection(self, conn_string: str):
        """Set up a persistent connection to DB"""
        return create_engine(conn_string)

    def close_db_connection(self):
        """Close DB connection"""
        self.db_connection.dispose()

    def _setup_direct_access(self):
        if self.metadata is None:
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.db_connection)
            self.table = self.metadata.tables.get(self.table_name)
            if self.table is None:
                raise ValueError(f"Table '{self.table_name}' not found in db")

    def get_session(self, sess_id: str):
        """Retrieve an existing chat session"""
        if sess_id is None:
            return None
        return SQLChatMessageHistory(
            session_id=sess_id,
            connection=self.db_connection,
            table_name=self.table_name,
            session_id_field_name=self.session_id_field_name,
        )

    def add_messages(self, sess_id: str, messages: List[str]):
        """Store new messages in a session"""
        if sess_id is None:
            return
        messages = merge_message_runs(messages, chunk_separator="")
        self.get_session(sess_id).add_messages(messages)

    def get_session_names(self) -> List:
        """Retrieves a unique list of session names from the DB"""
        try:
            self._setup_direct_access()
        except ValueError as e:
            self.logger.info(str(e))
            return []

        with Session(self.db_connection) as session:
            stmt: Select = select(distinct(self.table.c[self.session_id_field_name]))
            result = session.execute(stmt)

            session_ids = [row[0] for row in result]

        return session_ids


class VariableHandler:
    """Handles variable replacements in text"""

    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("app.variables")
        self.logger.addHandler(default_handler)
        self.logger.setLevel(logging.INFO)

    def resolve(self, template, variables: Dict):
        """Replace variables in template (text)"""
        self.logger.info("Variables replaced: %s", variables)
        return re.sub(
            r"{{\s*([^}\s]+)\s*}}",
            lambda match: str(variables.get(match.group(1), match.group(0))),
            template,
        )


class AnswerGenerator:
    """Handle API connections and text generations using LLM providers"""

    def __init__(self, config, smanager: Optional[SessionManager] = None):
        self.config = config
        self.session_manager = smanager

        self.logger = logging.getLogger("app.generator")
        self.logger.addHandler(default_handler)
        self.logger.setLevel(logging.INFO)

        self._clients: Dict[str, Any] = {}

    def _load_profile(self, profile_name):
        profile = self.config.get("profiles", {}).get(profile_name, None)
        if profile is None:
            raise ValueError(f"Profile {profile_name} not found in config")
        return profile

    def _get_profile(self, profile_name):
        if profile_name is None:  # try default profile
            profile_name = self.config.get("profiles", {}).get("default_profile", None)
            if profile_name is None:
                raise ValueError("No default profile defined")

        return profile_name, self._load_profile(profile_name)

    @staticmethod
    def _basic_auth(username, password):
        token = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _get_ollama_client(self, profile: Dict):
        username = profile.get("client", {}).get("username", None)
        password = profile.get("client", {}).get("password", None)
        host = profile.get("client", {}).get("url", None)
        headers = (
            {"Authorization": self._basic_auth(username, password)}
            if username and password
            else None
        )

        if host:
            return ollama.Client(host=host, headers=headers)

        raise ValueError("Ollama URL not defined in config")

    def _get_azure_openai_client(self, profile: Dict):
        endpoint = profile.get("azure_endpoint", None)
        api_key = profile.get("api_key", None)
        api_version = profile.get("api_version", "2024-08-01-preview")

        assert endpoint is not None, "Endpoint for profile not found in config!"
        assert api_key is not None, "API key for profile not found in config!"

        return openai.AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )

    def _get_openai_client(self, profile: Dict):
        api_key = profile.get("api_key", None)
        assert api_key is not None, "API key for profile not found in config!"
        return openai.OpenAI(api_key=api_key)

    def _get_client(self, profile_name):
        if profile_name in self._clients:
            return self._clients[profile_name]

        _, profile = self._get_profile(profile_name)
        profile_type = profile.get("type", None)

        if profile_type is None:
            raise ValueError("Profile type undefined")

        # TODO: outsource to modular system
        if profile["type"].lower() == "ollama":
            self._clients[profile_name] = self._get_ollama_client(profile)
        elif profile["type"].lower() == "azure_openai":
            self._clients[profile_name] = self._get_azure_openai_client(profile)
        elif profile["type"].lower() == "openai":
            self._clients[profile_name] = self._get_openai_client(profile)
        else:
            raise ValueError(f"Uknown profile type: {profile_type}")

        return self._clients[profile_name]

    def _generate_openai(
        self,
        _: str,
        model: str,
        messages: Iterable[Any],
        stream: bool = False,
        **kwargs,
    ):
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            **kwargs,
        )
        if stream:
            yield from response
        else:
            return response

    def _generate_azure_openai(
        self,
        profile_name: str,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        **kwargs,
    ):
        response = self._get_client(profile_name).chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            **kwargs,
        )
        if stream:
            yield from response
        else:
            yield response

    def _generate_ollama(
        self,
        profile_name: str,
        model: str,
        messages=List[Dict],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ):
        client = self._get_client(profile_name)
        response = client.generate(
            model=model,
            system=messages[0]["content"],
            prompt=messages[1]["content"],
            options=options,
            stream=stream,
        )
        if stream:
            yield from response
        else:
            yield response

    def translate_options_to_openai(self, options: Dict[str, Any]):
        """Translate keys in options dict to OpenAI compatible keys
        Anything not found in mappign will be ignored!
        """
        mapping = {  # "ollama_name": "openai_name"
            "temperature": "temperature",
            "num_predict": "max_completion_tokens",
            "top_p": "top_p",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }

        ignored = []
        translated = {}
        for key, val in options.items():
            if key in mapping:
                translated[mapping[key]] = val
            else:
                ignored.append(key)

        return translated, ignored

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o-mini"):
        """Count tokens of text"""
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)

    @staticmethod
    def _chatmessages_to_json(messages: List[ChatMessage]):
        """Convert langchain message format to JSON compatible with APIs
        This is needed, because of the non-langchain implementation of generate()
        """
        return [{"role": msg.type, "content": msg.content} for msg in messages]

    def generate(
        self,
        profile_name: str,
        model: str,
        messages: List[ChatMessage],
        options: Dict[str, Any],
        stream: bool,
        session: Optional[str] = None,
    ):
        """Main function, generates text based on messages"""
        api_messages = self._chatmessages_to_json(messages)
        profile_name, profile = self._get_profile(profile_name)

        service = profile.get("type", None)
        if service is None:
            raise ValueError("Service type missing from profile")

        if model is None:
            model = profile.get("default_model", None)
            if model is None:
                raise ValueError("Model unspecified")

        self.logger.info(
            "profile:%s // service:%s // model:%s // options:%s // stream:%s",
            profile_name,
            service,
            model,
            options,
            stream,
        )

        if service in ("openai", "azure_openai"):
            translated_options, ignored_options = self.translate_options_to_openai(
                options
            )
            self.logger.debug("Ignored options in the request: %s", ignored_options)

            if service == "openai":
                generate = self._generate_openai
            else:
                generate = self._generate_azure_openai

            def response_stream_openai():
                for chunk in generate(
                    profile_name,
                    model,
                    api_messages,
                    stream=stream,
                    **translated_options,
                ):
                    messages = []
                    if stream:
                        ret = {}
                        if len(chunk.choices):
                            if chunk.choices[0].delta.content is not None:
                                txt = chunk.choices[0].delta.content
                                ret["response"] = txt
                                messages.append(
                                    ChatMessageChunk(content=txt, role="ai")
                                )
                            if chunk.choices[0].finish_reason == "stop":
                                ret["last_chunk"] = True
                        if chunk.usage:
                            ret["usage"] = {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            }
                        yield json.dumps(ret) + "\n"
                    else:
                        txt = chunk.choices[0].message.content
                        messages.append(ChatMessage(content=txt, role="ai"))
                        yield json.dumps(
                            {
                                "response": txt,
                                "usage": {
                                    "prompt_tokens": chunk.usage.prompt_tokens,
                                    "completion_tokens": chunk.usage.completion_tokens,
                                    "total_tokens": chunk.usage.total_tokens,
                                },
                                "ignored_options": ",".join(ignored_options),
                            }
                        )
                self.session_manager.add_messages(session, messages)

            return Response(response_stream_openai(), content_type="application/json")

        if service == "ollama":

            def response_stream_ollama():
                messages = []
                for chunk in self._generate_ollama(
                    profile_name, model, api_messages, stream=stream, options=options
                ):
                    messages.append(ChatMessageChunk(content=chunk.response, role="ai"))
                    yield json.dumps({"response": chunk.response}) + "\n"
                self.session_manager.add_messages(session, messages)

            return Response(response_stream_ollama(), content_type="application/json")

        raise ValueError(f"Unknown service {service}")


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
        "-c", "--config", default="./config.json", help="Path to config JSON file"
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
