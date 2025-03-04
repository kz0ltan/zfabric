#!/usr/bin/env python3

import json
import os

from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth


ENV_PATH = "~/.config/zfabric/.env"


def get_spaces(instance_url: str, space_name: str, auth):
    headers = {"Accept": "application/json"}
    url = instance_url + "/wiki/api/v2/spaces"

    spaces = []
    while True:
        response = requests.request("GET", url, headers=headers, auth=auth, timeout=30)
        r_json = response.json()
        spaces.extend(r_json["results"])
        print(len(spaces), end="\r")
        for space in r_json["results"]:
            if space["name"] == space_name:
                print(json.dumps(space, indent=2))
        if "_links" in r_json and "next" in r_json["_links"]:
            url = instance_url + r_json["_links"]["next"]
        else:
            with open("/tmp/spaces", "w", encoding="utf-8") as fp:
                json.dump(spaces, fp)
            break


def get_pages(instance_url: str, space_id: int, auth):
    headers = {"Accept": "application/json"}
    url = instance_url + f"/wiki/api/v2/spaces/{space_id}/pages"

    response = requests.request("GET", url, headers=headers, auth=auth, timeout=30)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    load_dotenv(os.path.expanduser(ENV_PATH))
    instance_url = str(os.getenv("CONFLUENCE_URL"))
    token = str(os.getenv("CONFLUENCE_TOKEN"))
    user = str(os.getenv("CONFLUENCE_USER"))
    space_id = int(os.getenv("CONFLUENCE_SPACE"))

    auth = HTTPBasicAuth(user, token)
    get_pages(instance_url, space_id, auth)
