#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback

USER_ID=${LOCAL_USER_ID:-9001}
GROUP_ID=${LOCAL_GROUP_ID:-9001}

echo "Starting with UID : $USER_ID"
groupadd -f -g $GROUP_ID dockeruser
useradd --shell /bin/bash -u $USER_ID -g $GROUP_ID -o -c "" -m dockeruser
echo 'dockeruser:password123456' | chpasswd
export HOME=/home/dockeruser

/usr/bin/supervisord