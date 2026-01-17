import logging
import re
from typing import Dict

from flask.logging import default_handler

from .config import Config


class VariableHandler:
    """Handles variable replacements in text"""

    def __init__(self, config: Config):
        self.config = config

        self.logger = logging.getLogger("app.variables")
        self.logger.addHandler(default_handler)
        self.logger.setLevel(self.config.get("logging.loglevel", default=logging.INFO))

    def coalesce_data(self, data: Dict[str, str]):
        """Joins data values into one data["input"]"""
        if len(data) == 1 and list(data.keys())[0] == "input":
            return data["input"]

        if "input" not in data:
            data["input"] = ""

        for source in list(data.keys()):
            if source == "input":
                continue

            if len(data["input"]) == 0:
                data["input"] = data[source]
            else:
                data["input"] += "\n\n" + data[source]
            del data[source]

        return data["input"]

    def insert_data_into_template_with_globbing(self, template: str, data: Dict[str, str]) -> str:
        """
        Replace {{  }} expressions in template from data[] using regular expressions.
        Replaced keys should be deleted from data.
        """
        pattern = re.compile(r"{{\s*([\w\d_*?]+)\s*}}")

        def matches_pattern(key: str, pattern: str) -> bool:
            """Check if the key matches the glob pattern."""
            if pattern == "*":
                return True
            if not pattern:
                return False

            # Convert glob pattern to regex
            regex_pattern = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
            return re.match(regex_pattern, key) is not None

        def replace_match(match):
            # this returns only the matched key, without {{ and }}
            glob_key = match.group(1)
            matched_values = []

            # Find all keys that match the glob pattern and join
            for key in list(data.keys()):
                if matches_pattern(key, glob_key):
                    matched_values.append(data.pop(key))
                    self.logger.debug("Input data inserted: %s", key)

            # If we found any matches, return them joined
            if len(matched_values) > 0:
                return "\n\n".join(matched_values)

            # Return the original match if no keys matched
            return match.group(0)

        return re.sub(pattern, replace_match, template)

    def resolve(
        self,
        template: str,
        variables: Dict[str, str],
        data: Dict[str, str],
        coalesce_data: bool = False,
    ):
        """
        Replace variables in template (text)
        If <var_name> exists in variables, replace, otherwise leave {{ var_name }} in template
        """
        self.logger.debug("Variables replaced: %s", variables)
        template = re.sub(
            r"{{\s*([\w\d_]+)\s*}}",
            lambda match: str(variables.get(match.group(1), match.group(0))),
            template,
        )

        template = self.insert_data_into_template_with_globbing(template, data)

        if coalesce_data and len(data):
            return (
                template + "\n\n" + self.coalesce_data(data)
                if len(template)
                else self.coalesce_data(data)
            )

        return template
