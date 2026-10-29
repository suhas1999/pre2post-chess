#!/usr/bin/env bash
# Run on the laptop. Drives cloud/run_paper_setup.sh on an interruptible Vast box: starts it once
# setup is done, relaunches it after a preemption (verl resumes from the latest checkpoint), and
# when /workspace/ALL_DONE appears downloads everything except FSDP shards, verifies the HF
# checkpoints, and destroys the box. If the download cannot be verified it stops the box instead.
# Prints one line per event (start, step progress every 25 steps, pause, relaunch, done).
#
#   caffeinate -ims bash cloud/watch_interruptible.sh INSTANCE_ID LOCAL_DIR [TEST_SET_DIR]
set -uo pipefail

ID="$1"; LOCAL_DIR="$2"; TEST_SETS="${3:-}"
MAX_HOURS="${MAX_HOURS:-10}"; STALL_MIN="${STALL_MIN:-30}"
RUN_ENV="${RUN_ENV:-N_GPUS=4 SAVE_FREQ=10}"
mkdir -p "${LOCAL_DIR}"
LOG="${LOCAL_DIR}/watch.log"
say() { echo "$(date -u '+%H:%M') $*" | tee -a "${LOG}"; }

conn() {   # sets HOST PORT from the current ssh url; returns 1 if the box is unreachable
    local url; url="$(uvx vastai ssh-url "${ID}" 2>/dev/null)"
    [[ "${url}" == ssh://* ]] || return 1
    url="${url#ssh://root@}"; HOST="${url%:*}"; PORT="${url##*:}"
}
remote() { ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    -p "${PORT}" "root@${HOST}" "$@" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun"; }

launch() {
    if [ -n "${TEST_SETS}" ]; then
        remote "mkdir -p /workspace/data/puzzles"
        rsync -a -e "ssh -o BatchMode=yes -p ${PORT}" "${TEST_SETS}/" "root@${HOST}:/workspace/data/puzzles/" 2>>"${LOG}"
    fi
    remote "cd /workspace/pre2post-chess && git pull -q --ff-only; \
        (${RUN_ENV} setsid nohup bash cloud/run_paper_setup.sh >> /workspace/paper.log 2>&1 < /dev/null &)"
}

start=$(date +%s); last_bucket=-1; last_size=-1; last_change=${start}; was_down=0
say "watching instance ${ID}"
while true; do
    now=$(date +%s)
    if [ $((now - start)) -ge $((MAX_HOURS * 3600)) ]; then
        say "hit ${MAX_HOURS}h cap: stopping the box (disk kept)"; uvx vastai stop instance "${ID}" >>"${LOG}" 2>&1; exit 1
    fi
    status=$(uvx vastai show instance "${ID}" --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
    if [ "${status}" != "running" ] || ! conn; then
        [ ${was_down} -eq 0 ] && say "box not running (status=${status:-?}): paused or preempted, waiting"
        was_down=1; last_change=${now}; sleep 60; continue
    fi
    s=$(remote 'D=0; [ -e /workspace/ALL_DONE ] && D=1; F=0; [ -e /workspace/ALL_FAILED ] && F=1
        U=0; grep -q "Setup done" /workspace/setup.log 2>/dev/null && U=1
        P=0; pgrep -f "[r]un_paper_setup.sh" >/dev/null && P=1
        L=$(ls -t /workspace/rl_runs/*/logs/*.log 2>/dev/null | head -1)
        Z=$(cat "$L" /workspace/eval.log 2>/dev/null | wc -c)
        M=$(grep -oE "step:[0-9]+ .*" "$L" 2>/dev/null | tail -1 | tr " " "\n" | grep -E "^step:|^critic/score/mean:|^timing_s/step:|^response_length/mean:" | tr "\n" " ")
        echo "$D $F $U $P ${Z:-0} | $M"')
    [ -z "${s}" ] && { sleep 60; continue; }
    read -r done failed setup running size _ <<<"${s%%|*}"
    metrics="${s#*|}"
    if [ ${was_down} -eq 1 ]; then say "box is back"; was_down=0; fi
    if [ "${done}" = 1 ]; then say "run + eval finished"; break; fi
    if [ "${failed}" = 1 ]; then say "run FAILED: downloading what exists, stopping the box"; break; fi
    if [ "${setup}" = 1 ] && [ "${running}" = 0 ]; then
        say "starting run_paper_setup.sh (${RUN_ENV})"; launch; last_change=${now}; sleep 60; continue
    fi
    step=$(grep -oE 'step:[0-9]+' <<<"${metrics}" | cut -d: -f2)
    if [ -n "${step}" ] && [ $((step / 25)) -ne ${last_bucket} ]; then
        last_bucket=$((step / 25)); say "${metrics}"
    fi
    if [ "${size}" != "${last_size}" ]; then last_size=${size}; last_change=${now}; fi
    if [ "${running}" = 1 ] && [ $((now - last_change)) -ge $((STALL_MIN * 60)) ]; then
        say "no log output for ${STALL_MIN} min: restarting the run"
        remote 'for p in $(pgrep -f "[r]un_paper_setup.sh|[r]un_rl.sh|[m]ain_ppo|[e]val_rl.sh"); do kill $p; done; sleep 10; source /workspace/pre2post-chess/cloud/env.sh; ray stop --force'
        last_change=${now}
    fi
    sleep 60
done

say "downloading to ${LOCAL_DIR}"
verified=0
for attempt in 1 2 3; do
    rsync -a -e "ssh -o BatchMode=yes -p ${PORT}" --exclude='checkpoints/' "root@${HOST}:/workspace/rl_runs/" "${LOCAL_DIR}/rl_runs/" 2>>"${LOG}"
    rsync -a -e "ssh -o BatchMode=yes -p ${PORT}" "root@${HOST}:/workspace/eval_results/" "${LOCAL_DIR}/eval_results/" 2>>"${LOG}"
    rsync -a -e "ssh -o BatchMode=yes -p ${PORT}" "root@${HOST}:/workspace/paper.log" "root@${HOST}:/workspace/eval.log" "${LOCAL_DIR}/" 2>>"${LOG}"
    r=$(remote "cd /workspace/rl_runs && find . -path '*checkpoints_hf_format*' -name '*.safetensors' -printf '%p %s\n' | sort")
    l=$(cd "${LOCAL_DIR}/rl_runs" 2>/dev/null && find . -path '*checkpoints_hf_format*' -name '*.safetensors' -exec stat -f '%N %z' {} \; | sort)
    say "attempt ${attempt}: remote $(grep -c . <<<"${r}") HF ckpts, local $(grep -c . <<<"${l}")"
    if [ -n "${r}" ] && [ "${r}" = "${l}" ]; then verified=1; break; fi
    sleep 30
done
if [ ${verified} -ne 1 ] || [ "${failed}" = 1 ]; then
    say "not verified or run failed: STOPPING the box (disk kept)"; uvx vastai stop instance "${ID}" >>"${LOG}" 2>&1; exit 1
fi
say "verified; destroying ${ID}"
uvx vastai destroy instance "${ID}" -y >>"${LOG}" 2>&1
say "done"
