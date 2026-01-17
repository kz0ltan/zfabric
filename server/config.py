from typing import Dict, List

from pydantic import BaseModel

from server.lib.servicekit import Configuration


class DBConfig(BaseModel):
    path: str


class UserDetails(BaseModel):
    realname: str
    api_key: str


class Config(Configuration):
    pattern_paths: List[str]
    context_paths: List[str]
    db: DBConfig
    users: Dict[str, UserDetails]

    class Config:
        extra = "forbid"  # disallows extra keys not defined in the model
