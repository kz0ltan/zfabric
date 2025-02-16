#!/usr/bin/env python3

import argparse
from base64 import b64encode
import copy
import json
from functools import wraps, partial
import itertools
import logging
import os
from pathlib import Path
import re
import tiktoken
from typing import Dict, List, Any

from helpers import load_file

from flask import Flask, request, jsonify, Response
from flask.logging import default_handler
import ollama
import openai

class FabricAPIServer:

    def __init__(self, name:str="zFabric", config_path:str=None):
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH")
        self.config = json.loads(load_file(config_path))

        self.app = Flask(name)
        self.app.logger.setLevel(logging.INFO)
        self.add_routes()

        self.generator = AnswerGenerator(self.config)
        self.variable_handler = VariableHandler(self.config)

    def check_auth_token(self, token:str):
        db = self.config["users"]
        for user in db.keys():
            if db[user]["api_key"] == token:
                return user
        return None

    def auth_required(self, f):
        """ Decorator function to check if the token is valid.

        Args:
            f: The function to be decorated

        Returns:
            The decorated function
        """

        @wraps(f)
        def decorated_function(*args, **kwargs):
            """ Decorated function to handle authentication token and API endpoint.

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

            return f(*args, **kwargs)

        return decorated_function

    def add_errohandlers(self):
        @self.app.errorhandler(404)
        def not_found(e):
            return jsonify({"error": "The requested resource was not found."}), 404

        @self.app.errorhandler(500)
        def server_error(e):
            return jsonify({"error": "An internal server error occurred."}), 500

    def add_routes(self):

        @self.app.route("/profiles", methods=["GET"])
        @self.auth_required
        def list_profiles():
            try:
                profiles = [p for p in self.config["profiles"].keys()]
                if "default_profile" in profiles:
                    profiles.remove("default_profile")
                    default_profile = self.config["profiles"]["default_profile"]
                    profiles.insert(0, "Default: " + str(default_profile))
                return jsonify({"response": profiles})
            except Exception as e:
                self.app.logger.error(f"Error occured: {e}")
                return jsonify({"error": "An error occurred while processing the request."}), 500

        @self.app.route("/patterns", methods=["GET"])
        @self.auth_required
        def list_patterns():
            try:
                patterns = itertools.chain.from_iterable(
                        [os.listdir(ppath) for ppath in self.config["pattern_paths"]])
                return jsonify({"response": list(patterns)})
            except Exception as e:
                self.app.logger.error(f"Error occured: {e}")
                return jsonify({"error": "An error occurred while processing the request."}), 500

        @self.app.route("/patterns/<pattern>", methods=["POST"])
        @self.auth_required
        def milling(pattern):
            """ Combine fabric pattern with input from user and send to OpenAI's GPT-4 model.

            Returns:
                JSON: A JSON response containing the generated response or an error message.

            Raises:
                Exception: If there is an error during the API call.
            """

            profile_name = request.args.get("profile", None)
            model = request.args.get("model", None)
            options = request.args.get("options", None)
            stream = request.args.get("stream", default=False, type=(lambda s: s.lower() in ("True", "true", '1')))
            variables = request.args.get("variables", default=dict())

            if options:
                options = json.loads(options)

            data = request.get_json()
            if "input" not in data:
                return jsonify({"error": "Missing input parameter"}), 400

            input_data = data["input"]

            for ppath in self.config['pattern_paths']:
                pattern_path = Path(ppath) / pattern
                if pattern_path.exists():
                    break
                if ppath == self.config['pattern_paths'][-1]:
                    return jsonify({"error": "Pattern not found"}), 400

            system_prompt = load_file(pattern_path / "system.md", "")
            user_prompt = load_file(pattern_path / "user.md", "")

            system_prompt = self.variable_handler.resolve(system_prompt, variables)
            user_prompt = self.variable_handler.resolve(user_prompt, variables)

            # Build the API call
            system_message = {"role": "system", "content": system_prompt}
            user_message = {"role": "user", "content": user_prompt + "\n" + input_data}
            messages = [system_message, user_message]

            try:
                return self.generator.generate(profile_name, model, messages, options, stream)
            except Exception as e:
                self.app.logger.error(f"Error occured: {e}")
                raise e
                #return jsonify({"error": "An error occurred while processing the request."}), 500

class VariableHandler:

    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger('app.variables')
        self.logger.addHandler(default_handler)
        self.logger.setLevel(logging.INFO)

    def resolve(self, template, variables:Dict):
        self.logger.info(f"Variables replaced: {variables}")
        return re.sub(r'{{\s*([^}\s]+)\s*}}',
            lambda match: str(variables.get(match.group(1), match.group(0))),
            template
        )

class AnswerGenerator:

    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger('app.generator')
        self.logger.addHandler(default_handler)
        self.logger.setLevel(logging.INFO)

        self._clients = dict()

    def _load_profile(self, profile_name):
        profile = self.config.get("profiles", {}).get(profile_name, None)
        if profile is None:
            raise ValueError("Profile {} not found in config".format(profile_name))
        return profile

    def _getProfile(self, profile_name):
        if profile_name is None: # try default profile
            profile_name = self.config.get("profiles", {}).get("default_profile", None)
            if profile_name is None:
                raise ValueError("No default profile defined")

        return profile_name, self._load_profile(profile_name)

    @staticmethod
    def _basic_auth(username, password):
        token = b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")
        return f'Basic {token}'

    def _getOllamaClient(self, profile:Dict):
        username = profile.get("client", {}).get("username", None)
        password = profile.get("client", {}).get("password", None)
        host = profile.get("client", {}).get("url", None)
        headers = {"Authorization": self._basic_auth(username, password)
            } if username and password else None

        if host:
            return ollama.Client(host=host, headers=headers)
        else:
            raise ValueError("Ollama URL not defined in config")

    def _getAzureOpenAIClient(self, profile:Dict):
        endpoint = profile.get("azure_endpoint", None)
        api_key = profile.get("api_key", None)
        api_version = profile.get("api_version", "2024-08-01-preview")

        assert endpoint is not None, "Endpoint for profile not found in config!"
        assert api_key is not None, "API key for profile not found in config!"

        return openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )

    def _getOpenAIClient(self, profile:Dict):
        api_key = profile.get("api_key", None)
        assert api_key is not None, "API key for profile not found in config!"
        return openai.OpenAI(api_key)

    def _getClient(self, profile_name):
        if profile_name in self._clients:
            return self._clients[profile_name]
        else:
            _, profile = self._getProfile(profile_name)
            profile_type = profile.get("type", None)

            if profile_type is None:
                raise ValueError("Profile type undefined")
            # TODO: outsource to modular system
            elif profile["type"].lower() == "ollama":
                self._clients[profile_name] = self._getOllamaClient(profile)
            elif profile["type"].lower() == "azure_openai":
                self._clients[profile_name] = self._getAzureOpenAIClient(profile)
            elif profile["type"].lower() == "openai":
                self.clients[profile_name] = self._getOpenAIClient(profile)
            else:
                raise ValueError("Uknown profile type: {}".format(profile_type))

            return self._clients[profile_name]

    def _generate_openai(self, profile_name:str, model:str, messages:List[Dict], stream:bool=False, **kwargs):
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            **kwargs
        )
        if stream:
            yield from response
        else:
            return response

    def _generate_azure_openai(self, profile_name:str, model:str, messages:List[Dict], stream:bool=False, **kwargs):
        response = self._getClient(profile_name).chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            **kwargs
        )
        if stream:
            yield from response
        else:
            yield response

    def _generate_ollama(self, profile_name:str, model:str, messages=List[Dict], stream:bool=False, options:Dict[str,Any]=None):
        client = self._getClient(profile_name)
        response = client.generate(
                model=model,
                system=messages[0]["content"],
                prompt=messages[1]["content"],
                options=options,
                stream=stream
        )
        if stream:
            yield from response
        else:
            yield response

    def translate_options_to_openai(self, options:Dict[str,Any]):
        """ Translate keys in options dict to OpenAI compatible keys
            Anything not found in mappign will be ignored!
        """
        mapping = { # "ollama_name": "openai_name"
            "temperature": "temperature",
            "num_predict": "max_completion_tokens",
            "top_p": "top_p",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }

        ignored = list()
        translated = dict()
        for k, v in options.items():
            if k in mapping:
                translated[mapping[k]] = v
            else:
                ignored.append(k)

        return translated, ignored

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o-mini"):
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)

    def generate(self, profile_name, model, messages, options, stream):
        profile_name, profile = self._getProfile(profile_name)

        service = profile.get("type", None)
        if service is None:
            raise ValueError("Service type missing from profile")

        if model is None:
            model = profile.get("default_model", None)
            if model is None:
                raise ValueError("Model unspecified")

        self.logger.info(f"profile:{profile_name} // service:{service} // model:{model} // options:{options} // stream:{stream}")

        if service in ("openai", "azure_openai"):

            translated_options, ignored_options = self.translate_options_to_openai(options)
            self.logger.debug(f"Ignored options in the request: {ignored_options}")

            if service == "openai":
                generate = self.generate_openai
            else:
                generate = self._generate_azure_openai

            def response_stream():
                for chunk in generate(profile_name, model, messages, stream=stream, **translated_options):
                    if stream:
                        r = dict()
                        if len(chunk.choices):
                            if chunk.choices[0].delta.content is not None:
                                r["response"] = chunk.choices[0].delta.content
                            if chunk.choices[0].finish_reason == "stop":
                                r["last_chunk"] = True
                        if chunk.usage:
                            r["usage"] = {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            }
                        yield json.dumps(r) + '\n'
                    else:
                        text = chunk.choices[0].message.content
                        yield json.dumps({
                            "response": chunk.choices[0].message.content,
                            "usage": {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            },
                            "ignored_options": ','.join(ignored_options)
                        })

            return Response(response_stream(), content_type="application/json")
        elif service == "ollama":
            def response_stream():
                for chunk in self._generate_ollama(profile_name, model, messages, stream=stream, options=options):
                    yield json.dumps({"response": chunk.response}) + "\n"
            return Response(response_stream(), content_type="application/json")
        else:
            raise ValueError("Unknown service {}".format(service))

def parse_arguments():
    parser = argparse.ArgumentParser(description="zFabric server application")

    parser.add_argument('-l', '--listen', type=str, default="localhost",
            help="Hostname/IP to listen on")
    parser.add_argument('-p', '--port', type=int, default=13337,
            help="Port to listen on")
    parser.add_argument('-d', '--debug', action="store_true", default=False,
            help="Werkzeug debug mode")
    parser.add_argument('-c', '--config', default="./config.json",
            help="Path to config JSON file")

    return parser.parse_args()

def start():
    """ Meant to be used by Gunicorn
    """
    return FabricAPIServer().app

if __name__ == "__main__":
    args = parse_arguments()
    FabricAPIServer("zFabric", args.config).app.run(host=args.listen, port=args.port, debug=args.debug)
