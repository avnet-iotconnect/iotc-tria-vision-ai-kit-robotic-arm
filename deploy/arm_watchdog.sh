#!/bin/sh
# Boot + recovery supervisor for the robotic-arm pick-and-place demo.
#
# Launches the app in IDLE: the arm connects and IOTCONNECT + the camera stack
# come up, but the arm DOES NOT MOVE and no vision runs until an operator sends
# a start command from the cloud. This keeps the arm safe on the table at
# power-up and after any hiccup.
#
#   START movement:  set_mode  mode=yolo-pickplace     (from IOTCONNECT)
#   STOP  movement:  set_mode  mode=idle   (or: stop_demo)
#
# The supervisor also: relaunches (into idle) when the xArm re-enumerates after
# a USB drop, and relaunches (into idle) if the app dies while the arm is
# healthy. Both recoveries return to the SAFE idle state, not to moving.
#
# Pause the supervisor (e.g. to calibrate over SSH):  touch /tmp/demo_watchdog_off
# Resume:                                              rm /tmp/demo_watchdog_off
BASE=/root/iotc-tria-vision-ai-kit-robotic-arm
LOG=/tmp/watchdog.log
CONF=0.75          # YOLO confidence used once yolo-pickplace is started; edit to retune per venue
STARTUP_GRACE=75   # seconds to let start.sh (conda activate takes ~60s on a cold board) bring
                   # main.py up before the health check runs — prevents duplicate launches

arm_present() { lsusb | grep -q 0483:5750; }
demo_running() { pgrep -f 'python -u main.py' >/dev/null; }
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

wait_network() {
    # Wait for DNS to resolve the IOTCONNECT host so the app connects to the
    # cloud at launch. Without this, a cold boot on a slow-DHCP network starts
    # the app before DNS is ready -> IOTCONNECT connect fails -> the app runs
    # "without cloud connectivity" and dashboard set_mode commands never arrive.
    # Up to ~180s, then launch anyway (a cloud-less demo beats no demo). Venue
    # DHCP/DNS has been seen to take >90s after a cold boot.
    i=0
    while [ "$i" -lt 60 ]; do
        if getent hosts discovery.iotconnect.io >/dev/null 2>&1; then
            log "network ready (DNS resolves IOTCONNECT)"
            return 0
        fi
        i=$((i + 1)); sleep 3
    done
    log "network NOT ready after 180s -> launching cloud-less (restart service once network is up)"
}

launch_idle() {
    wait_network
    cd "$BASE" || return
    YOLO_CONF="$CONF" nohup ./start.sh --mode idle --web-port 8080 \
        > /tmp/yolo.log 2>&1 < /dev/null &
    log "launched app in IDLE (send set_mode mode=yolo-pickplace to start movement)"
}

was_present=0
dead_since=0
log "supervisor started (idle-launch, CONF=$CONF)"
while true; do
    if [ -f /tmp/demo_watchdog_off ]; then sleep 10; continue; fi
    if arm_present; then
        if [ "$was_present" = 0 ]; then
            log "arm present (boot or replug) -> (re)launching in idle"
            pkill -f 'python -u main.py'; sleep 3
            pkill -9 -f 'python -u main.py' 2>/dev/null; sleep 1
            launch_idle
            sleep "$STARTUP_GRACE"
        elif ! demo_running; then
            now=$(date +%s)
            [ "$dead_since" = 0 ] && dead_since=$now
            if [ $((now - dead_since)) -ge "$STARTUP_GRACE" ]; then
                log "app not running with healthy arm -> launching idle"
                launch_idle
                dead_since=0
                sleep "$STARTUP_GRACE"
            fi
        else
            dead_since=0
        fi
        was_present=1
    else
        was_present=0
        dead_since=0
    fi
    sleep 5
done
