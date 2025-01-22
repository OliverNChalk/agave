#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH_CONFIG_ARGS=(-o StrictHostKeyChecking=no)

declare -A nodes
if [ -f /home/sol/invalidator/.env ]; then
    # shellcheck disable=SC1091
    source /home/sol/invalidator/.env
    for i in {0..9}; do
        ip=${IPS[$i]}
        pubkey=${PUBKEYS[$i]:0:4}
        nodes["$pubkey"]="$ip"
    done
else
    echo "/home/sol/invalidator/.env file not found. please ensure you deploy the cluster with the latest infra-invalidator" >&2
    exit 1
fi

# stop all agave-validator
echo "Stopping agave-validator on all nodes..."
for name in "${!nodes[@]}"; do
    ip="${nodes[$name]}"

    # rotate log to make sure the log is not too old
    ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "sudo logrotate -f /etc/logrotate.d/sol"

    if ! ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "pidof agave-validator" >/dev/null; then
        echo "not found agave-validator process, ($name, $ip). skip..."
        continue
    else
        echo "found agave-validator process, ($name, $ip). killing..."
        ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "pkill agave-validator"
    fi
done

# setting all nodes into wen-restart mode
echo "Setting all nodes into wen-restart mode..."

# update validator scripts
echo "Updating validator scripts"
for name in "${!nodes[@]}"; do
    ip="${nodes[$name]}"
    target_path=/home/sol/invalidator/scripts/update-validator-scripts-to-wen-restart-mode.sh
    scp -p "$here/update-validator-scripts-to-wen-restart-mode.sh" "sol@$ip":"$target_path"
    # shellcheck disable=SC2029
    ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "$target_path"
done

# HACK: restart bootstrap node first because we're using it as the entrypoint
echo "Restarting bootstrap validators"
ssh "${SSH_CONFIG_ARGS[@]}" sol@"${nodes[tiv1]}" /home/sol/invalidator/scripts/bootstrap-validator.sh

echo "Waiting 10s for bootstrap node startup"
sleep 10

echo "Restarting other validators..."
for name in "${!nodes[@]}"; do
    if [[ $name == "tiv1" ]]; then
        continue
    fi
    ip="${nodes[$name]}"
    ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" /home/sol/invalidator/scripts/validator.sh
done

echo "Waiting for all nodes to see wen restart completion"
attempt=0
while true; do
    found_nodes=0

    for name in "${!nodes[@]}"; do
        ip="${nodes[$name]}"
        if ssh "${SSH_CONFIG_ARGS[@]}" sol@"$ip" "grep 'Wen start finished' /home/sol/invalidator/logs/agave-validator.log > /dev/null 2>&1"; then
            found_nodes=$((found_nodes + 1))
        fi
    done

    if [ $found_nodes -eq ${#nodes[@]} ]; then
        echo "All nodes have seen wen restart completion"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -ge 60 ]; then
        echo >&2 "Not all nodes see wen restart completion within 10 minutes. A manual check is needed."
        exit 1
    fi

    echo "Found wen restart completion in $found_nodes/${#nodes[@]} nodes. Waiting 10s..."
    sleep 10
done

# parse proto result from tiv1
echo "Fetching and parsing proto result from tiv1..."
ip="${nodes[tiv1]}"
target_path=/home/sol/invalidator/scripts/parse-wen-restart.sh
scp -p "$here/parse-wen-restart.sh" "sol@$ip":"$target_path"
# shellcheck disable=SC2029
result=$(ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "$target_path")

slot="$(echo "$result" | jq -r '.slot')"
bank_hash="$(echo "$result" | jq -r '.bankhash')"
shred_version="$(echo "$result" | jq -r '.shred_version')"

# restarting cluster
echo "Restarting cluster..."

echo "Restarting bootstrap validator..."
ip="${nodes[tiv1]}"
target_path=/home/sol/invalidator/scripts/update-validator-scripts-to-wait-for-supermajority.sh
scp -p "$here/update-validator-scripts-to-wait-for-supermajority.sh" "sol@$ip":"$target_path"
ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "pkill agave-validator" # bootstrap validator won't be killed by the wen restart script. do it manually.
# shellcheck disable=SC2029
ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "BANK_HASH=$bank_hash SHRED_VERSION=$shred_version SLOT=$slot $target_path"
ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" /home/sol/invalidator/scripts/bootstrap-validator.sh

echo "Waiting 60s for bootstrap node startup"
sleep 60

echo "Restarting other validators..."
for name in "${!nodes[@]}"; do
    if [[ $name == "tiv1" ]]; then
        continue
    fi
    ip="${nodes[$name]}"

    target_path=/home/sol/invalidator/scripts/update-validator-scripts-to-wait-for-supermajority.sh
    scp -p "$here/update-validator-scripts-to-wait-for-supermajority.sh" "sol@$ip":"$target_path"
    # shellcheck disable=SC2029
    ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" "BANK_HASH=\"$bank_hash\" SHRED_VERSION=\"$shred_version\" SLOT=\"$slot\" $target_path"
    ssh "${SSH_CONFIG_ARGS[@]}" "sol@$ip" /home/sol/invalidator/scripts/validator.sh
done

echo "Waiting for all nodes to see new root"
attempt=0
while true; do
    found_nodes=0

    for name in "${!nodes[@]}"; do
        ip="${nodes[$name]}"
        if ssh "${SSH_CONFIG_ARGS[@]}" sol@"$ip" "grep 'new root' /home/sol/invalidator/logs/agave-validator.log > /dev/null 2>&1"; then
            found_nodes=$((found_nodes + 1))
        fi
    done

    if [ $found_nodes -eq ${#nodes[@]} ]; then
        echo "All nodes have seen new root completion"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -ge 60 ]; then
        echo >&2 "Not all nodes see new root within 10 minutes. A manual check is needed."
        exit 1
    fi

    echo "Found new root in $found_nodes/${#nodes[@]} nodes. Waiting 10s..."
    sleep 10
done
