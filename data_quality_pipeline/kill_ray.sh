#!bin/bash

ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs kill -9