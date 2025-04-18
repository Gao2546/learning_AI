#!/bin/bash

if [!command -v "sudo" &> /dev/null]; then
    apt-get install sudo || sudo apt-get update || sudo apt-get upgrade
fi

if [!command -v "psql" &> /dev/null]; then
    sudo apt-get install postgresql
fi

sudo -u postgres psql -c "CREATE USER athip WITH PASSWORD '123456';"