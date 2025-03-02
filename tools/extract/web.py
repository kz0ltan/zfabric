#!/usr/bin/env python3

import argparse
import os

from dotenv import load_dotenv
import requests

ENV_PATH = "~/.config/zfabric/.env"

# https://stackoverflow.com/questions/4672060/web-scraping-how-to-identify-main-content-on-a-webpage
# https://github.com/scrapinghub/article-extraction-benchmark
# https://github.com/goose3/goose3


def extract(url: str, lib: str) -> str:
    """Extract content from url using lib"""
    if lib == "newspaper3k":
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    if lib == "readability-lxml":
        from readability import Document
        response = requests.get(url, timeout=10)
        doc = Document(response.content)
        return doc.summary()

    if lib == "jina.ai":
        headers = {
            "Authorization": "Bearer " + str(os.getenv("JINA_TOKEN")),
        }
        response = requests.get("https://r.jina.ai/" +
                                url, headers=headers, timeout=10)
        return response.text

    return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CLI tool to extract content using libraries')
    parser.add_argument('url', help='Content URL')
    parser.add_argument(
        '--lib', choices=["newspaper3k", "readability-lxml", "jina.ai"], default="jina.ai")

    args = parser.parse_args()
    load_dotenv(os.path.expanduser(ENV_PATH))

    print(extract(args.url, args.lib))
