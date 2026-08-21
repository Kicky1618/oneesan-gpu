#!/bin/bash
set -e
mkdir -p /root/.ssh
chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
grep -q 'oneesan-vast-debug' /root/.ssh/authorized_keys || cat >> /root/.ssh/authorized_keys <<'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID8X7vdHoD3J5IaNzdhlte9RFMAtf5O5xlonLooVPBnB oneesan-vast-debug
EOF
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync build-essential >/tmp/oneesan-apt.log 2>&1 || true
