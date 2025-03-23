#!/usr/bin/env python3

# https://joplinapp.org/help/api/references/rest_api#post-notes

# Get dir (parent) ids:
#curl http://localhost:41184/notes?token=<token>

# POST note
#curl --data '{"parent_id": "<parent_id>",  "title": "Test Note", "body_html": "Some note in **Markdown**"}' http://127.0.0.1:41184/notes?token=<token>
