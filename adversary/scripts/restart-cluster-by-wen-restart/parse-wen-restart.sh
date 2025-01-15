#!/usr/bin/env bash

set -o errexit

if [[ -z $WEN_RESTART_FILE_PATH ]]; then
  WEN_RESTART_FILE_PATH=/home/sol/invalidator/wen-restart
fi

# HACK: should use the same version of validators, but the latest version is used here.
curl -s -o /tmp/wen_restart.proto https://raw.githubusercontent.com/anza-xyz/agave/master/wen-restart/proto/wen_restart.proto

result="$(protoc \
  --experimental_allow_proto3_optional \
  --decode=solana.wen_restart_proto.WenRestartProgress \
  --proto_path /tmp/ \
  wen_restart.proto <$WEN_RESTART_FILE_PATH)"

my_snapshot=$(echo "$result" | sed -n '/my_snapshot {/,/}/p')

path=$(echo "$my_snapshot" | grep 'path:' | sed 's/.*path: "\(.*\)".*/\1/')
slot=$(echo "$my_snapshot" | grep 'slot:' | sed 's/.*slot: \(.*\)/\1/')
bankhash=$(echo "$my_snapshot" | grep 'bankhash:' | sed 's/.*bankhash: "\(.*\)".*/\1/')
shred_version=$(echo "$my_snapshot" | grep 'shred_version:' | sed 's/.*shred_version: \(.*\)/\1/')

cat <<EOF
{
  "path": "$path",
  "slot": "$slot",
  "bankhash": "$bankhash",
  "shred_version": "$shred_version"
}
EOF
