from functools import wraps

from flask import request, jsonify


def auth_required(server):
    """Decorator function to check if the request's auth token is valid.

    Args:
        f: The function to be decorated

    Returns:
        The decorated function
    """

    def decorator(func):
        @wraps(func)
        def decorated_function(*args, **kwargs):
            """Decorated function to handle authentication token and API endpoint.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Result of the decorated function.

            Raises:
                KeyError: If 'Authorization' header is not found in the request.
                TypeError: If 'Authorization' header value is not a string.
                ValueError: If the authentication token is invalid or expired.
            """

            if server is None:
                raise ValueError("Server instance is not initialized")

            # Get the authentication token from request header
            auth_token = request.headers.get("Authorization", "")

            # Remove any bearer token prefix if present
            if auth_token.lower().startswith("bearer "):
                auth_token = auth_token[7:]

            # Check if token is valid
            user = server.check_auth_token(auth_token)
            if user is None:
                return jsonify({"error": "user not found!"}), 401

            return func(*args, **kwargs)

        return decorated_function

    return decorator
