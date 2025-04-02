import logging
import os
import re
from typing import Dict, List, Any

from flask.logging import default_handler
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages.utils import merge_message_runs
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine, select, distinct, MetaData, delete
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from server.helpers import ensure_directories_exist
from server.lib.langchain import ExMessageConverter


class SessionManager:
    """
    Handles sessions (single query or chat)
    * Save/delete/append sessions using configured storage mechanism
    * Works with langchain to store / retrieve sessions
    """

    def __init__(self, config: Dict[Any, Any]):
        self.db_path = config.get(
            "sqlite3_db_path", "~/.local/share/zfabric/sessions.sqlite3")
        self.table_name = config.get("sqlite3_table_name", "message_store")
        self.session_id_field_name = config.get(
            "sqlite3_session_id_field_name", "session_id")
        self._db_connection = None
        self.metadata = None
        self.table = None

        self.config = config

        self.logger = logging.getLogger("app.variables")
        self.logger.addHandler(default_handler)
        self.logger.setLevel(self.config.get("loglevel", logging.INFO))

    @property
    def db_connection(self):
        """Lazy setup of db connection"""
        if self._db_connection:
            return self._db_connection
        if len(self.db_path) == 0:
            self.logger.warning(
                "No 'sqlite3_conn_string' was found in config, using volatile, memory storage!"
            )
        else:
            ensure_directories_exist(self.db_path)
            self.db_path = "sqlite:///" + os.path.expanduser(self.db_path)
        self._db_connection = self._setup_db_connection(self.db_path)
        return self._db_connection

    def _setup_db_connection(self, conn_string: str):
        """Set up a persistent connection to DB"""
        self.logger.info(f"Opening DB file {conn_string}")
        return create_engine(conn_string)

    def close_db_connection(self):
        """Close DB connection"""
        self.db_connection.dispose()

    def _setup_direct_access(self):
        # check if reflection from the existing DB file has happened before or not
        if self.table is None:
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
            custom_message_converter=ExMessageConverter(self.table_name),
        )

    def add_messages(self, sess_id: str, messages: List[ChatMessage], merge: bool = True):
        """Store new messages in a session"""
        if sess_id is None:
            return
        if merge:
            messages = merge_message_runs(messages, chunk_separator="")
        self.get_session(sess_id).add_messages(messages)

    def get_session_names(self) -> List:
        """Retrieves a unique list of session names from the DB"""
        try:
            self._setup_direct_access()
        except ValueError as ex:
            self.logger.info(str(ex))
            return []

        with Session(self.db_connection) as session:
            stmt: Select = select(
                distinct(self.table.c[self.session_id_field_name]))
            result = session.execute(stmt)

            session_ids = [row[0] for row in result]

        return session_ids

    def delete_session(self, sess_id: str):
        """Deletes all messages from a specific session"""
        try:
            self._setup_direct_access()
        except ValueError as ex:
            self.logger.info(str(ex))
            return False

        with Session(self.db_connection) as session:
            stmt = delete(self.table)

            if sess_id != "all":
                stmt = stmt.where(
                    self.table.c[self.session_id_field_name] == sess_id)
            result = session.execute(stmt)
            session.commit()

            if result.rowcount > 0:
                self.logger.info(
                    f"Deleted {result.rowcount} messages from session: {sess_id}")
            else:
                self.logger.info(f"No messages found in session: {sess_id}")

        return True


class VariableHandler:
    """Handles variable replacements in text"""

    def __init__(self, config):
        self.config = config

        self.logger = logging.getLogger("app.variables")
        self.logger.addHandler(default_handler)
        self.logger.setLevel(self.config.get("loglevel", logging.INFO))

    def coalesce_data(self, data: Dict[str, str]):
        if len(data) == 1:
            return data["input"]

        for source in list(data.keys()):
            if source == "input":
                continue

            if len(data["input"]) == 0:
                data["input"] = data[source]
            else:
                data["input"] += "\n\n" + data[source]
            del data[source]

        return data["input"]

    def insert_data_into_template(self, template: str, data: Dict[str, str]) -> str:
        """
        Replace {{  }} expressions in template from data[] using regular expressions.
        Replaced keys should be deleted from data.
        """
        pattern = re.compile(r"{{\s*([\w\d_]+)\s*}}")

        def replace_match(match):
            key = match.group(1)
            if key in data:
                self.logger.debug("Input data inserted: %s", key)
                return data.pop(key)
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

        template = self.insert_data_into_template(template, data)

        if coalesce_data and len(data):
            return (
                template + "\n\n" + self.coalesce_data(data)
                if len(template)
                else self.coalesce_data(data)
            )

        return template
