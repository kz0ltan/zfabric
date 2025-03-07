"""Extension for Langchain SQLite message history to be able to store time in DB"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_community.chat_message_histories.sql import BaseMessageConverter
from sqlalchemy import Column, Integer, Text

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base


def create_message_model(table_name: str, DynamicBase: Any) -> Any:
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.

    """

    # Model declared inside a function to have a dynamic table name.
    class Message(DynamicBase):  # type: ignore[valid-type, misc]
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        timestamp = Column(Integer)
        session_id = Column(Text)
        message = Column(Text)

    return Message


class ExMessageConverter(BaseMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, declarative_base())

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        return self.model_class(
            session_id=session_id,
            message=json.dumps(message_to_dict(message)),
            timestamp=message.response_metadata["timestamp"],
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class
