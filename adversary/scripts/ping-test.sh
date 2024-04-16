# shellcheck disable=SC2148
#
# Testing latency by pinging the network under different configurations and
# activity levels
#

for requiredVar in metricsWriteDatapoint runtime sleeptime solanaClient; do
  if ! declare -p "$requiredVar" >/dev/null; then
    cat <<EOM
ping-test expects $requiredVar to be set.  Functions defined in this
file are defined with an assumption that this variable is a global variable
defined by the script that includes this script via the 'source' command.
EOM
    exit 1
  fi
done

parse_and_report_output() {
  local output="$1"
  local measurementName="$2"

  # Initialize fields array
  local -A fields

  # Extract transactions submitted and confirmed
  fields[transactions_submitted]=$(echo "$output" | grep -oP '(\d+) transactions submitted' | grep -oP '\d+')
  fields[transactions_confirmed]=$(echo "$output" | grep -oP '(\d+) transactions confirmed' | grep -oP '\d+')

  # Extract latency fields only if there are more than 0 transactions submitted
  if [[ "${fields[transactions_submitted]}" -gt 0 ]]; then
    fields[min_latency]=$(echo "$output" | grep -oP 'min/mean/max/stddev = \d+' | grep -oP '\d+')
    fields[mean_latency]=$(echo "$output" | grep -oP 'min/mean/max/stddev = \d+/\K\d+' | head -n 1)
    fields[max_latency]=$(echo "$output" | grep -oP 'min/mean/max/stddev = \d+/\d+/(\K\d+)' | head -n 1)
    # We only calculate the standard deviation if more than 1 transaction was confirmed
    if [[ "${fields[transactions_confirmed]}" -gt 1 ]]; then
      fields[stddev_latency]=$(echo "$output" | grep -oP 'min/mean/max/stddev = \d+/\d+/\d+/(\K\d+)' | head -n 1)
    else
      fields[stddev_latency]=0
    fi
  else
    # Set latency fields to 0 if no transactions were submitted
    fields[min_latency]=0
    fields[mean_latency]=0
    fields[max_latency]=0
    fields[stddev_latency]=0
  fi

  # Construct a comma-separated string of fields
  local fields_string
  for key in "${!fields[@]}"; do
    fields_string+="${key}=${fields[$key]},"
  done

  # Remove the trailing comma
  fields_string="${fields_string%,}"

  # shellcheck disable=SC2154
  $metricsWriteDatapoint "$measurementName,hostname=$HOSTNAME ${fields_string}"
}

test_ping() {
  # Indicate this test is running
  $metricsWriteDatapoint "adversary,hostname=$HOSTNAME test-ping=1"
  # Airdrop to ensure the client has enough funds to send transactions
  # shellcheck disable=SC2154
  "$solanaClient" -ul airdrop 1
  local ping_output
  # shellcheck disable=SC2154
  ping_output=$(timeout "$runtime" "$solanaClient" -ul ping 2>&1) || true
  parse_and_report_output "$ping_output" "adversary-test-ping"
  # Indicate this test is no longer running
  $metricsWriteDatapoint "adversary,hostname=$HOSTNAME test-ping=0"
  # shellcheck disable=SC2154
  sleep "$sleeptime"
}

test_pingWithPriority() {
  # Indicate this test is running
  $metricsWriteDatapoint "adversary,hostname=$HOSTNAME test-ping=2"
  # Airdrop to ensure the client has enough funds to send transactions
  "$solanaClient" -ul airdrop 1
  local ping_output_with_cu
  ping_output_with_cu=$(timeout "$runtime" "$solanaClient" -ul ping --with-compute-unit-price 1 2>&1) || true
  parse_and_report_output "$ping_output_with_cu" "adversary-test-ping-with-priority"
  # Indicate this test is no longer running
  $metricsWriteDatapoint "adversary,hostname=$HOSTNAME test-ping=0"
  sleep "$sleeptime"
}