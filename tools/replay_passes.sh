#!/bin/sh
# Replay data/<dir> as satellite passes, then exit: the base, ephemerides and rover files of
# each pass (B_/N_/R_ with the same index) are sent at once, followed by a wait long enough for
# the receiver to close the window (IDLE_TIMEOUT) and process it. Runs in the `launcher`
# service of docker-compose.yml: `docker compose run --rm launcher [dir]`, dir "." (the sample
# capture, default) or "drone".
set -e
DIR=${1:-.}
WAIT=${WAIT:-90}
send() {
    python /tools/generic_simulator_files.py --dir "/data/$DIR" --prefix "$1" --ip "${TARGET:-server}" \
        --port "$2" --once --index "$3"
}
PASSES=$(ls /data/"$DIR"/R_*.rtcm3 | wc -l)
for i in $(seq 1 "$PASSES"); do
    echo "Pass $i/$PASSES"
    send B "${PORT_BASE:-5005}" "$i"
    send N "${PORT_NAV:-5007}" "$i"
    send R "${PORT_ROVER:-5006}" "$i"
    echo "Waiting ${WAIT} s for the receiver to process it"
    sleep "$WAIT"
done
echo "Done: results in logs/server_execution.log (ntn_server) and logs/anomaly/server.log"
