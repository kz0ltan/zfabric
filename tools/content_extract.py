#!/usr/bin/env python3

import argparse
import os

from dotenv import load_dotenv

ENV_PATH="~/.config/zfabric/.env"

def extract(url:str, lib:str):
    if lib == "newspaper3k":
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    elif lib == "readability-lxml":
        import requests
        from readability import Document
        response = requests.get(url)
        doc = Document(response.content)
        return doc.summary()
    elif lib == "jina.ai":
        import requests
        headers = {
            'Authorization': 'Bearer ' + os.getenv("JINA_TOKEN"),
        }
        response = requests.get("https://r.jina.ai/" + url, headers=headers)
        return response.text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI tool to extract content using libraries')
    parser.add_argument('url', help='Content URL')
    parser.add_argument('--lib', choices=["newspaper3k", "readability-lxml", "jina.ai"], default="jina.ai")

    args = parser.parse_args()
    load_dotenv(os.path.expanduser(ENV_PATH))

    print(extract(args.url, args.lib))
