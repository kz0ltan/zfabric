#!/usr/bin/env python3

"""Download comments from Reddit"""

import praw

reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="Fabric",
    username="jlibling",
    password="",
)


def get_all_comments(post_url: str):
    """Fetch all comments from a Reddit post"""
    submission = reddit.submission(url=post_url)
    submission.comments.replace_more(limit=None)  # Load all nested comments

    comments = []
    for comment in submission.comments.list():
        comments.append(comment.body)

    return comments


# Example usage
url = "https://www.reddit.com/r/singularity/comments/1j1rejq/the_past_18_months_have_seen_the_most_rapid/"
all_comments = get_all_comments(url)

# Print or save the comments
for i, comment in enumerate(all_comments, 1):
    print(f"{i}. {comment}\n")
