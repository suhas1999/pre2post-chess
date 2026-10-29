#!/usr/bin/env bash
# Run on the laptop. Waits for a remote cloud/run_rl.sh run to end, downloads the results,
# verifies them, then destroys the instance. Never destroys unverified data: if the
# download cannot be verified it STOPS the instance instead (disk kept, GPU billing ends).
#
#   caffeinate -ims bash cloud/finish_and_destroy.sh INSTANCE_ID HOST PORT REMOTE_RUN_DIR LOCAL_DIR
#
# REMOTE_RUN_DIR is ${SAVE_DIR}/${EXPERIMENT_NAME} on the box. Also ends the run early
# (keeping the saved checkpoints) after MAX_HOURS, or when the training log has not grown
# for STALL_MIN minutes. Partial results are synced down every SYNC_MIN minutes.
set -uo pipefail

ID="$1"; HOST="$2"; PORT="$3"; REMOTE_DIR="$4"; LOCAL_DIR="$5"
MAX_HOURS="${MAX_HOURS:-5.5}"
STALL_MIN="${STALL_MIN:-30}"
SYNC_MIN="${SYNC_MIN:-15}"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -p "${PORT}" "root@${HOST}")
RSYNC_SSH="ssh -o ConnectTimeout=20 -p ${PORT}"
mkdir -p "${LOCAL_DIR}"
LOG="${LOCAL_DIR}/auto_finish.log"
log() { echo "$(date '+%F %T') $*" | tee -a "${LOG}"; }

remote() { "${SSH[@]}" "$@" 2>/dev/null | grep -v "Welcome to vast.ai\|Have fun"; }

# FSDP shards (checkpoints/) stay on the box; the HF copies come down.
sync_down() {
    rsync -a -e "${RSYNC_SSH}" --exclude='checkpoints/' "root@${HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/" 2>>"${LOG}"
    rsync -a -e "${RSYNC_SSH}" "root@${HOST}:/workspace/rl.log" "${LOCAL_DIR}/" 2>>"${LOG}"
}

max_secs=$(awk "BEGIN{print int(${MAX_HOURS}*3600)}")
start=$(date +%s); last_sync=${start}; last_size=-1; last_change=${start}; ssh_fail=0; reason=""
log "watching instance ${ID} (${HOST}:${PORT}) ${REMOTE_DIR}"
while true; do
    out=$(remote "if [ -e ${REMOTE_DIR}/DONE ]; then echo DONE; elif [ -e ${REMOTE_DIR}/FAILED ]; then echo FAILED; else echo RUN; fi; \
        stat -c %s /workspace/rl.log 2>/dev/null || echo 0; grep -o 'step:[0-9]*' /workspace/rl.log | tail -1")
    if [ -z "${out}" ]; then
        ssh_fail=$((ssh_fail + 1)); log "ssh failed (${ssh_fail})"
        if [ ${ssh_fail} -ge 30 ]; then reason="ssh unreachable"; break; fi
        sleep 60; continue
    fi
    ssh_fail=0
    state=$(echo "${out}" | sed -n 1p); size=$(echo "${out}" | sed -n 2p); step=$(echo "${out}" | sed -n 3p)
    now=$(date +%s)
    if [ "${size}" != "${last_size}" ]; then last_size=${size}; last_change=${now}; fi
    if [ $((now - last_sync)) -ge $((SYNC_MIN * 60)) ]; then
        log "state=${state} ${step}"; sync_down; last_sync=${now}
    fi
    if [ "${state}" != RUN ]; then reason="run ${state}"; break; fi
    if [ $((now - start)) -ge ${max_secs} ]; then reason="hit ${MAX_HOURS}h cap"; break; fi
    if [ $((now - last_change)) -ge $((STALL_MIN * 60)) ]; then reason="stalled ${STALL_MIN} min"; break; fi
    sleep 60
done
log "ending: ${reason}"

if [ "${reason}" = "ssh unreachable" ]; then
    log "cannot reach the box; STOPPING it (data kept on disk)"
    uvx vastai stop instance "${ID}" 2>&1 | tee -a "${LOG}"
    exit 1
fi

if [ "${reason}" != "run DONE" ] && [ "${reason}" != "run FAILED" ]; then
    log "stopping the run early and converting the saved checkpoints"
    remote "pkill -f '[m]ain_ppo'; sleep 15; ray stop --force; pkill -9 -f 'VLLM::EngineCore'; \
        cd /workspace/pre2post-chess && source cloud/env.sh && cd rl/verl && \
        LARGE_DIR=\${SAVE_DIR} bash 8_gpu_bash/convert_to_hf_all.sh" | tail -6 | tee -a "${LOG}"
fi

# Verified = every HF checkpoint on the box is here with the same weight-file size.
list_cmd="cd ${REMOTE_DIR} 2>/dev/null && find checkpoints_hf_format -name '*.safetensors' -printf '%p %s\n' 2>/dev/null | sort"
verified=0
for attempt in 1 2 3 4 5; do
    sync_down
    remote_list=$(remote "${list_cmd}")
    local_list=$(cd "${LOCAL_DIR}" && find checkpoints_hf_format -name '*.safetensors' -exec stat -f '%N %z' {} \; 2>/dev/null | sort)
    log "attempt ${attempt}: remote $(echo -n "${remote_list}" | grep -c .) HF ckpts, local $(echo -n "${local_list}" | grep -c .)"
    if [ -n "${remote_list}" ] && [ "${remote_list}" = "${local_list}" ] && [ -f "${LOCAL_DIR}/rl.log" ]; then
        verified=1; break
    fi
    sleep 30
done

if [ ${verified} -ne 1 ]; then
    log "download NOT verified; STOPPING the instance instead of destroying (data kept)"
    uvx vastai stop instance "${ID}" 2>&1 | tee -a "${LOG}"
    exit 1
fi

log "verified download:"; ls -la "${LOCAL_DIR}" "${LOCAL_DIR}/checkpoints_hf_format" | tee -a "${LOG}"
log "destroying instance ${ID}"
uvx vastai destroy instance "${ID}" -y 2>&1 | tee -a "${LOG}"
uvx vastai show instances 2>&1 | tail -3 | tee -a "${LOG}"
log "done"
