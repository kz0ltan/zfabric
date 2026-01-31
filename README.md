# zFabric

This is an alternative implementation of [Fabric](https://github.com/danielmiessler/fabric) by Daniel Miessler.

The tool yt.py was directly copied from Fabric (and modified since).

## Installation

### Dependencies

To download repo::

    git clone https://github.com/kz0ltan/zfabric
    cd zfabric

To install the required libraries::

    uv sync

### Server

Copy config.json example::

    cp examples/config.json .

Edit config.json as you wish, for example add a random token for your user, add API key(s), etc.

To run the server on [Mac OS as a service](https://gist.github.com/johndturn/09a5c055e6a56ab61212204607940fa0#further-reading):

    cp examples/hu.kzoltan.zFabric.plist ~/Library/LaunchAgents/
    ...edit service file
    launchctl load ~/Library/LaunchAgents/hu.kzoltan.zFabric.plist
    launchctl enable user/<uid>/hu.kzoltan.zFabric
    launchctl start hu.kzoltan.zFabric

Check if Gunicorn is running as intended:

    launchctl list | grep hu.kzoltan.zFabric

You should see a PID in the first column. If you see '-' there, check logs/error.log

### client

Edit ~/.config/zfabric/.env::

    ZF_SERVER_TOKEN=<token>

## Uninstall

To remove the service::

    launchctl disable user/<uid>/hu.kzoltan.zFabric
    launchctl unload ~/Library/LaunchAgents/hu.kzoltan.zFabric.plist

Delete the repository, and the client config file from ~/.config/zFabric/.env

## Development

To run the local server::

    ./app.py -d

Running workspace dependencies::

    uv run -m tools.extract.reddit -h
    uv run -m tools.extract.web -h
    uv run -m open_deep_research
