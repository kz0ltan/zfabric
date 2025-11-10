import base64
import copy
import datetime
import json
import logging
from typing import Optional, Dict, List, Any, Iterable, Union

from anthropic import Anthropic
from flask.logging import default_handler
from flask import Response
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.utils import get_buffer_string
from groq import Groq
import ollama
import openai
import tiktoken

from .models import SessionManager
from .helpers import merge_dicts


class Generator:
    """Handle API connections and text generations using LLM providers"""

    def __init__(self, config, smanager: Optional[SessionManager] = None):
        self.config = config
        self.session_manager = smanager

        self.logger = logging.getLogger("app.generator")
        self.logger.addHandler(default_handler)
        self.logger.setLevel(self.config.get("loglevel", logging.INFO))

        self._clients: Dict[str, Any] = {}

    def _load_profile(self, profile_name):
        profile = self.config.get("profiles", {}).get(profile_name, None)
        if profile is None:
            raise ValueError(f"Profile '{profile_name}' not found in config")
        return profile

    def _get_profile(self, profile_name):
        if profile_name is None:  # try default profile
            profile_name = self.config.get(
                "profiles", {}).get("default_profile", None)
            if profile_name is None:
                raise ValueError("No default profile defined")

        return profile_name, self._load_profile(profile_name)

    @staticmethod
    def _basic_auth(username, password):
        token = base64.b64encode(
            f"{username}:{password}".encode("utf-8")).decode("ascii")
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
        endpoint = profile.get("endpoint", None)
        api_key = profile.get("api_key", None)
        api_version = profile.get("api_version", "2024-08-01-preview")

        assert endpoint is not None, "Endpoint for profile not found in config!"
        assert api_key is not None, "API key for profile not found in config!"

        return openai.AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    def _get_openai_client(self, profile: Dict):
        base_url = profile.get("url", None)
        api_key = profile.get("api_key", None)
        assert api_key is not None, "API key for profile not found in config!"
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def _get_groq_client(self, profile: Dict):
        api_key = profile.get("api_key", None)
        assert api_key is not None, "API key for profile not found in config!"
        return Groq(api_key=api_key)

    def _get_anthropic_client(self, profile: Dict):
        api_key = profile.get("api_key", None)
        endpoint = profile.get("endpoint", None)
        assert api_key is not None, "API key for profile not found in config!"
        return Anthropic(api_key=api_key, base_url=endpoint, default_headers={"api-key": api_key})

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
            self._clients[profile_name] = self._get_azure_openai_client(
                profile)
        elif profile["type"].lower() == "openai":
            self._clients[profile_name] = self._get_openai_client(profile)
        elif profile["type"].lower() == "groq":
            self._clients[profile_name] = self._get_groq_client(profile)
        elif profile["type"].lower() == "anthropic":
            self._clients[profile_name] = self._get_anthropic_client(profile)
        else:
            raise ValueError(f"Uknown profile type: '{profile_type}'")

        return self._clients[profile_name]

    def _generate_openai(
        self,
        profile_name: str,
        model: str,
        messages: Iterable[Any],
        stream: bool = False,
        **kwargs,
    ):
        client = self._get_client(profile_name)
        response = client.chat.completions.create(
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

    def _generate_azure_openai(
        self,
        profile_name: str,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        **kwargs,
    ):
        client = self._get_client(profile_name)
        response = client.chat.completions.create(
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

    def _generate_groq(
        self,
        profile_name: str,
        model: str,
        messages=List[Dict],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ):
        client = self._get_client(profile_name)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            # stream_options={"include_usage": True} if stream else None,
            **options,
        )
        if stream:
            yield from response
        else:
            yield response

    def _generate_anthropic(
        self,
        profile_name: str,
        model: str,
        messages=List[Dict],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ):
        client = self._get_client(profile_name)

        system = None
        # Anthropic accepts system messages in a different way
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages.pop(0)

        response = client.messages.create(
            model=model, messages=messages, system=system, stream=stream, **options
        )

        if stream:
            yield from response
        else:
            yield response

    def _generate_ollama(
        self,
        profile_name: str,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[float, str]] = None,
    ):
        self.logger.debug(
            f"Ollama client.chat(model={model}, messages=..., "
            + f"options={options}, stream={stream}, keep_alive={keep_alive})"
        )
        client = self._get_client(profile_name)
        response = client.chat(
            model=model,
            messages=messages,
            options=options,
            stream=stream,
            keep_alive=keep_alive,
        )
        if stream:
            yield from response
        else:
            yield response

    def translate_options(self, options: Dict[str, Any], flavor: str = "openai"):
        """Translate keys in options dict to OpenAI compatible keys
        Anything not found in mappign will be ignored!
        """
        mapping = {}
        if flavor == "openai":
            mapping = {  # "ollama_name": "openai_name"
                "temperature": "temperature",
                "num_predict": "max_completion_tokens",
                "top_p": "top_p",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
            }
        elif flavor == "anthropic":
            mapping = {  # "ollama_name": "anthropic_name"
                "temperature": "temperature",
                "num_predict": "max_tokens",
                "top_p": "top_p",
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
    def count_tokens(text: str, model: str = "gpt-4o-mini", round_up=False):
        """Count tokens of text"""
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        n = len(tokens)

        if not round_up:
            return n
        else:
            return ((n + 8191) // 8192) * 8192

    @staticmethod
    def _chatmessages_to_json(messages: List[ChatMessage]):
        """Convert langchain message format to JSON compatible with APIs
        This is needed, because of the non-langchain implementation of generate()
        """
        return [{"role": msg.role, "content": copy.copy(msg.content)} for msg in messages]

    @staticmethod
    def _ollama_image_transformation(msg_list: List[Dict[str, Any]]):
        """
        Moves images (based64 encoded) out of the message list,
        so they can be fed to Ollama API
        """
        for i, message in enumerate(msg_list):
            if isinstance(message["content"], list):
                idx = 0
                txt = ""
                while idx < len(message["content"]):
                    if message["content"][idx]["type"] == "image_url":
                        if "images" not in message:
                            message["images"] = []
                        img = message["content"].pop(idx)
                        i = img["image_url"]["url"][23:]
                        message["images"].append(i)
                    elif message["content"][idx]["type"] == "text":
                        txt += message["content"].pop(idx)["text"]
                    else:
                        idx += 1
                msg_list[i]["content"] = txt
        return msg_list

    @staticmethod
    def _groq_image_transformation(messages: List[Dict[str, Any]]):
        """
        Moves system message to the first user message, as Groq does not
        support system messages with images
        """
        for message in messages:
            if isinstance(message["content"], list):
                if any(msg["type"] == "image_url" for msg in message["content"]):
                    if messages[0]["role"] == "system":
                        if messages[1]["role"] == "user":
                            messages[1]["content"].insert(
                                0, {"type": "text",
                                    "text": messages[0]["content"]}
                            )
                            messages.pop(0)

        return messages

    @staticmethod
    def _anthropic_image_transformation(messages: List[Dict[str, Any]]):
        """Convert image attachments to Anthropic format"""
        for message in messages:
            if isinstance(message["content"], list):
                for msg in message["content"]:
                    if msg["type"] == "image_url":
                        msg["type"] = "image"
                        msg["source"] = {"type": "base64"}
                        data = base64.b64decode(msg["image_url"]["url"][23:])
                        del msg["image_url"]
                        msg["source"]["data"] = base64.standard_b64encode(
                            data).decode("utf-8")
                        msg["source"]["media_type"] = "image/jpeg"

        return messages

    def generate(
        self,
        profile_name: str,
        messages: List[ChatMessage],
        options: Dict[str, Any],
        stream: bool,
        keep_alive: Optional[Union[float, str]] = None,
        model: Optional[str] = None,
        session: Optional[str] = None,
        warnings: List[str] = None,
    ):
        """Main function, generates text based on messages"""
        if not warnings:
            warnings = []

        api_messages = self._chatmessages_to_json(messages)
        profile_name, profile = self._get_profile(profile_name)

        service = profile.get("type", None)
        if service is None:
            raise ValueError("Service type missing from profile")

        if model is None:
            model = profile.get("default_model", None)
            if model is None:
                raise ValueError("Model unspecified")

        profile_options = profile.get("options", None)
        if profile_options is not None:
            options = merge_dicts(options, profile_options)

        self.logger.info(
            "profile:%s // service:%s // model:%s // options:%s // stream:%s",
            profile_name,
            service,
            model,
            options,
            stream,
        )

        timestamp = datetime.datetime.now().timestamp()

        if service in ("openai", "azure_openai"):
            translated_options, ignored_options = self.translate_options(
                options)
            if len(ignored_options):
                ignored_message = f"Ignored options in the request: {ignored_options}"
                self.logger.debug(ignored_message)
                warnings.append(ignored_message)

            if service == "openai":
                generate = self._generate_openai
            else:
                generate = self._generate_azure_openai

            def response_stream_openai():
                messages = []
                for chunk in generate(
                    profile_name,
                    model,
                    api_messages,
                    stream=stream,
                    **translated_options,
                ):
                    if stream:
                        ret = {"session": session}
                        if len(chunk.choices):
                            if chunk.choices[0].delta.content is not None:
                                txt = chunk.choices[0].delta.content
                                ret["response"] = txt
                                messages.append(
                                    ChatMessageChunk(
                                        content=txt,
                                        role="assistant",
                                        response_metadata={
                                            "model": model,
                                            "options": translated_options,
                                            "session": session,
                                            "timestamp": timestamp,
                                        },
                                    )
                                )
                            if chunk.choices[0].finish_reason == "stop":
                                ret["last_chunk"] = True
                        if len(warnings):
                            ret["warnings"] = copy.copy(warnings)
                            warnings.clear()
                        if chunk.usage:
                            ret["usage"] = {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            }
                        yield json.dumps(ret) + "\n"
                    else:
                        txt = chunk.choices[0].message.content
                        messages.append(
                            ChatMessage(
                                content=txt,
                                role="assistant",
                                response_metadata={
                                    "model": model,
                                    "options": translated_options,
                                    "session": session,
                                    "timestamp": timestamp,
                                },
                            )
                        )
                        yield json.dumps({
                            "response": txt,
                            "usage": {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            },
                            "ignored_options": ",".join(ignored_options),
                            "session": session,
                            "warnings": copy.copy(warnings),
                        })
                self.session_manager.add_messages(session, messages)

            return StreamingResponse(
                response_stream_openai(), self.logger, warnings, content_type="application/json"
            )

        if service == "groq":
            api_messages = self._groq_image_transformation(api_messages)
            translated_options, ignored_options = self.translate_options(
                options)
            if len(ignored_options):
                ignored_message = f"Ignored options in the request: {ignored_options}"
                self.logger.debug(ignored_message)
                warnings.append(ignored_message)

            def response_stream_groq():
                messages = []
                for chunk in self._generate_groq(
                    profile_name,
                    model,
                    api_messages,
                    stream=stream,
                    options=translated_options,
                ):
                    if stream:
                        ret = {"session": session}
                        if len(chunk.choices):
                            if chunk.choices[0].delta.content is not None:
                                txt = chunk.choices[0].delta.content
                                if len(txt):
                                    ret["response"] = txt
                                    messages.append(
                                        ChatMessageChunk(
                                            content=txt,
                                            role="assistant",
                                            response_metadata={
                                                "model": model,
                                                "options": translated_options,
                                                "session": session,
                                                "timestamp": timestamp,
                                            },
                                        )
                                    )
                            if chunk.choices[0].finish_reason == "stop":
                                ret["last_chunk"] = True

                        if len(warnings):
                            ret["warnings"] = copy.copy(warnings)
                            warnings.clear()

                        if chunk.choices[0].finish_reason:
                            ret["usage"] = {
                                "prompt_tokens": chunk.x_groq.usage.prompt_tokens,
                                "completion_tokens": chunk.x_groq.usage.completion_tokens,
                                "total_tokens": chunk.x_groq.usage.total_tokens,
                            }
                        yield json.dumps(ret) + "\n"
                    else:
                        txt = chunk.choices[0].message.content
                        messages.append(
                            ChatMessage(
                                content=txt,
                                role="assistant",
                                response_metadata={
                                    "model": model,
                                    "options": translated_options,
                                    "session": session,
                                    "timestamp": timestamp,
                                },
                            )
                        )
                        yield json.dumps({
                            "response": txt,
                            "usage": {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            },
                            "ignored_options": ",".join(ignored_options),
                            "session": session,
                            "warnings": copy.copy(warnings),
                        })
                self.session_manager.add_messages(session, messages)

            return StreamingResponse(
                response_stream_groq(), self.logger, warnings, content_type="application/json"
            )

        if service == "anthropic":
            api_messages = self._anthropic_image_transformation(api_messages)
            translated_options, ignored_options = self.translate_options(
                options, flavor="anthropic"
            )
            if len(ignored_options):
                ignored_message = f"Ignored options in the request: {ignored_options}"
                self.logger.debug(ignored_message)
                warnings.append(ignored_message)

            if "max_tokens" not in translated_options:
                translated_options["max_tokens"] = 4096
                warnings.append("max_tokens set to default 4096")

            def response_stream_anthropic():
                messages = []
                if stream:
                    for chunk in self._generate_anthropic(
                        profile_name,
                        model,
                        api_messages,
                        stream=stream,
                        options=translated_options,
                    ):
                        usage = {}
                        ret = {"response": "", "session": session}
                        if chunk.type == "content_block_delta":
                            ret["response"] = chunk.delta.text
                            messages.append(
                                ChatMessageChunk(
                                    content=chunk.delta.text,
                                    role="assistant",
                                    response_metadata={
                                        "model": model,
                                        "options": translated_options,
                                        "session": session,
                                        "timestamp": timestamp,
                                    },
                                )
                            )

                        elif chunk.type == "message_start":
                            usage["cache_creation_input_tokens"] = (
                                chunk.message.usage.cache_creation_input_tokens
                            )
                            usage["cache_read_input_tokens"] = (
                                chunk.message.usage.cache_read_input_tokens
                            )
                            usage["input_tokens"] = chunk.message.usage.input_tokens
                            usage["output_tokens"] = chunk.message.usage.output_tokens

                        elif chunk.type == "message_delta":
                            usage["output_tokens"] += chunk.usage.output_tokens

                        if len(warnings):
                            ret["warnings"] = copy.copy(warnings)
                            warnings.clear()

                        if len(ret["response"]) > 0:
                            yield json.dumps(ret) + "\n"
                            ret["response"] = ""

                        elif chunk.type == "message_stop":
                            del ret["response"]
                            ret["usage"] = usage
                            yield json.dumps(ret) + "\n"

                else:
                    for chunk in self._generate_anthropic(
                        profile_name,
                        model,
                        api_messages,
                        stream=stream,
                        options=translated_options,
                    ):
                        ret = {"response": "", "usage": {}, "session": session}
                        if chunk.type == "message":
                            ret["response"] += chunk.content[0].text
                            messages.append(
                                ChatMessageChunk(
                                    content=chunk.content[0].text,
                                    role="assistant",
                                    response_metadata={
                                        "model": model,
                                        "options": translated_options,
                                        "session": session,
                                        "timestamp": timestamp,
                                    },
                                )
                            )
                            ret["usage"]["cache_creation_input_tokens"] = (
                                chunk.usage.cache_creation_input_tokens
                            )
                            ret["usage"]["cache_read_input_tokens"] = (
                                chunk.usage.cache_read_input_tokens
                            )
                            ret["usage"]["input_tokens"] = chunk.usage.input_tokens
                            ret["usage"]["output_tokens"] = chunk.usage.output_tokens

                    ret["warnings"] = copy.copy(warnings)

                    yield json.dumps(ret) + "\n"

                self.session_manager.add_messages(session, messages)

            return StreamingResponse(
                response_stream_anthropic(), self.logger, warnings, content_type="application/json"
            )

        if service == "ollama":
            api_messages = self._ollama_image_transformation(api_messages)

            # setting context length for ollama dynamically forces ollama to
            # reload the model if a different ctx size is provided :/
            # The API does not allow to query loaded model's context size

            if "num_ctx" not in options or options["num_ctx"] is None:
                warnings.append(
                    "num_ctx not set, falling back to Ollama default!")
            else:
                input_ctx_len = self.count_tokens(get_buffer_string(messages))
                if options["num_ctx"] <= input_ctx_len * 1.2:
                    warnings.append(
                        f"num_ctx {options['num_ctx']} seems low "
                        f"compared to input context length of {input_ctx_len}"
                    )

            def response_stream_ollama():
                messages = []
                for chunk in self._generate_ollama(
                    profile_name,
                    model,
                    api_messages,
                    stream=stream,
                    options=options,
                    keep_alive=keep_alive,
                ):
                    messages.append(
                        ChatMessageChunk(
                            content=chunk.message.content,
                            role="assistant",
                            response_metadata={
                                "model": model,
                                "options": options,
                                "session": session,
                                "timestamp": timestamp,
                            },
                        )
                    )

                    r = {"response": chunk.message.content, "session": session}
                    if len(warnings):
                        r["warnings"] = copy.copy(warnings)
                        warnings.clear()

                    yield (json.dumps(r) + "\n")
                self.session_manager.add_messages(session, messages)

            return StreamingResponse(
                response_stream_ollama(), self.logger, warnings, content_type="application/json"
            )

        raise ValueError(f"Unknown service '{service}'")


class StreamingResponse(Response):
    """Extension of Response to catch exceptions happening in generators
    Normally these happen in the context of the WSGI server,
    so Flask's error handlers are not catching exceptions returned by Ollama for example
    """

    def __init__(self, generator, logger, warnings, *args, **kwargs):
        self.generator = generator
        self.logger = logger
        self.warnings = warnings
        super().__init__(self._stream(), *args, **kwargs)

    def _stream(self):
        try:
            yield from self.generator
        except Exception as e:
            self.logger.error(f"Error requesting inference server: {str(e)}")
            yield json.dumps({
                "error": f"Error requesting inference server: {str(e)}",
                "warnings": self.warnings,
            })
