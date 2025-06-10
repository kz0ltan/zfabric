#!/bin/bash

SERVICE="hu.kzoltan.zFabric"

case "$1" in
    start)
        echo "Starting $SERVICE..."
        launchctl start $SERVICE
        ;;
    enable)
        echo "Enabling $SERVICE..."
        launchctl enable $SERVICE
        ;;
    stop)
        echo "Stopping $SERVICE..."
        launchctl stop $SERVICE
        ;;
    disable)
        echo "Disabling $SERVICE..."
        launchctl disable $SERVICE
        ;;
    restart)
        echo "Restarting $SERVICE..."
        launchctl stop $SERVICE
        sleep 1
        launchctl start $SERVICE
        ;;
    status)
        echo "Checking status of $SERVICE..."
        launchctl list | grep $SERVICE
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
