#!/usr/bin/env python3

import argparse
import json
import os
from typing import Optional, List, Dict

from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth


ENV_PATH = "~/.config/zfabric/.env"


class ConfluenceClient:
    """
    Simple client able to authenticate to Confluence using a personal access token and retrieve:
    * pages of a space
    * page contents
    """

    def __init__(self, space_id: Optional[int] = None):
        self.user = str(os.getenv("CONFLUENCE_USER"))
        self.token = str(os.getenv("CONFLUENCE_TOKEN"))

        self.instance_url = str(os.getenv("CONFLUENCE_URL"))
        self.space_id = int(space_id or os.getenv("CONFLUENCE_SPACE"))

        self._auth = False

    @property
    def auth(self) -> HTTPBasicAuth:
        """Returns a BasicAuth object for requests"""
        if not self._auth:
            self._auth = HTTPBasicAuth(self.user, self.token)
        return self._auth

    def _send_request(self, url):
        headers = {"Accept": "application/json"}
        return requests.request("GET", url, headers=headers, auth=self.auth, timeout=30)

    def _paginated_request(
        self,
        url: str,
        max_pages: int = 1,
        search_field: Optional[str] = None,
        search_value: Optional[str] = None,
    ) -> List:
        page_idx = 0
        objects = []
        while True:
            response = self._send_request(url)
            r_json = response.json()
            objects.extend(r_json["results"])
            print(len(objects), end="\r")
            page_idx += 1
            if search_field and search_value:
                for obj in r_json["results"]:
                    if obj[search_field] == search_value:
                        return [obj]
            if (
                "_links" in r_json
                and "next" in r_json["_links"]
                and page_idx < max_pages
            ):
                url = self.instance_url + r_json["_links"]["next"]
            else:
                print("\n", end="")
                break

        return objects

    def get_spaces(self, space_name: Optional[str] = None) -> List:
        """
        Returns all space's metadata OR
        Return one confluence space's metadata with the name space_name
        """
        url = self.instance_url + "/wiki/api/v2/spaces"
        spaces = self._paginated_request(
            url, max_pages=99999, search_field="name", search_value=space_name
        )
        return spaces

    def get_pages_of_space(
        self, space_id: int, page_title: Optional[str] = None
    ) -> List:
        """Return all pages' metadata within a space"""
        url = self.instance_url + f"/wiki/api/v2/spaces/{space_id}/pages"
        pages = self._paginated_request(
            url, max_pages=99999, search_field="title", search_value=page_title
        )
        return pages

    def get_page_by_id(self, page_id: int) -> Dict:
        """Get page contents based on page_id"""
        url = self.instance_url + f"/wiki/api/v2/pages/{page_id}?body-format=view"
        response = self._send_request(url)
        return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple tool to download confluence pages"
    )
    parser.add_argument(
        "--page-id", required=True, type=int, help="Page ID of a confluence page"
    )
    args = parser.parse_args()

    load_dotenv(os.path.expanduser(ENV_PATH))
    client = ConfluenceClient()
    page = client.get_page_by_id(args.page_id)
    print(json.dumps(page, indent=2))
