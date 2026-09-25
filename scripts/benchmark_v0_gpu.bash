#!/usr/bin/env bash
set -eo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <mono|stereo> <cpu|opencv|ceres|both> [bag_name] [d435i|cuav-forward|cuav-45deg]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="$1"
COMPUTE="$2"
BAG_NAME="${3:-Outdoor_08}"
PROFILE="${4:-d435i}"
BAG="${FWAF_DATA_ROOT:-${ROOT}/data/ROS2/FWAF-VID}/${BAG_NAME}"
CONFIG_DIR="${ROOT}/config/FWAF-VID"
RESULT_ROOT="${VINS_BENCHMARK_RESULT_ROOT:-${ROOT}/results/v0_gpu_benchmark}"

case "${PROFILE}" in
  d435i) CONFIG_PREFIX="d435i_imu"; IMU_TOPIC="/camera/imu" ;;
  cuav-forward) CONFIG_PREFIX="cuav_imu_forward"; IMU_TOPIC="/mavros/imu/data" ;;
  cuav-45deg) CONFIG_PREFIX="cuav_imu_45deg"; IMU_TOPIC="/mavros/imu/data" ;;
  *) echo "Unsupported profile: ${PROFILE}" >&2; exit 2 ;;
esac

case "${MODE}" in
  mono) BASE_CONFIG="${CONFIG_DIR}/${CONFIG_PREFIX}_mono.yaml" ;;
  stereo) BASE_CONFIG="${CONFIG_DIR}/${CONFIG_PREFIX}_stereo.yaml" ;;
  *) echo "Unsupported mode: ${MODE}" >&2; exit 2 ;;
esac

case "${COMPUTE}" in
  cpu) USE_GPU=0; USE_GPU_FLOW=0; USE_GPU_CERES=0 ;;
  opencv) USE_GPU=1; USE_GPU_FLOW=1; USE_GPU_CERES=0 ;;
  ceres) USE_GPU=0; USE_GPU_FLOW=0; USE_GPU_CERES=1 ;;
  both) USE_GPU=1; USE_GPU_FLOW=1; USE_GPU_CERES=1 ;;
  *) echo "Unsupported compute mode: ${COMPUTE}" >&2; exit 2 ;;
esac

if [ ! -f "${BAG}/metadata.yaml" ]; then
  echo "ROS 2 bag not found: ${BAG}" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [ -n "${VINS_BENCHMARK_RUN_DIR:-}" ]; then
  RUN_DIR="${VINS_BENCHMARK_RUN_DIR}"
  mkdir -p "$(dirname "${RUN_DIR}")"
  mkdir "${RUN_DIR}" || { echo "Run directory already exists: ${RUN_DIR}" >&2; exit 2; }
else
  mkdir -p "${RESULT_ROOT}/${BAG_NAME}/${PROFILE}/${MODE}/${COMPUTE}"
  RUN_DIR="$(mktemp -d "${RESULT_ROOT}/${BAG_NAME}/${PROFILE}/${MODE}/${COMPUTE}/${STAMP}_XXXXXX")"
fi
mkdir -p "${RUN_DIR}/pose_graph"
echo "RUN_DIR=${RUN_DIR}" >&2
EFFECTIVE_CONFIG="${RUN_DIR}/effective_config.yaml"
cp "${BASE_CONFIG}" "${EFFECTIVE_CONFIG}"
cp "${CONFIG_DIR}/left.yaml" "${RUN_DIR}/left.yaml"
cp "${CONFIG_DIR}/right.yaml" "${RUN_DIR}/right.yaml"
SED_ARGS=(
  -e "s|^use_gpu:.*|use_gpu: ${USE_GPU}|"
  -e "s|^use_gpu_acc_flow:.*|use_gpu_acc_flow: ${USE_GPU_FLOW}|"
  -e "s|^use_gpu_ceres:.*|use_gpu_ceres: ${USE_GPU_CERES}|"
  -e 's|^show_track:.*|show_track: 0|'
  -e 's|^save_image:.*|save_image: 0|'
  -e "s|^output_path:.*|output_path: \"${RUN_DIR}\"|"
  -e "s|^pose_graph_save_path:.*|pose_graph_save_path: \"${RUN_DIR}/pose_graph\"|"
)
case "${VINS_DIAGNOSTICS:-1}" in
  0|1) DIAGNOSTICS="${VINS_DIAGNOSTICS:-1}" ;;
  *) echo 'VINS_DIAGNOSTICS must be 0 or 1' >&2; exit 2 ;;
esac
# These calibration references are relative to the original configuration, not the staged copy.
for KEY in kalibr_camchain imu_allan kalibr_imu; do
  VALUE="$(sed -n -E "s|^${KEY}:[[:space:]]*\"([^\"]+)\".*|\1|p" "${BASE_CONFIG}")"
  if [ -n "${VALUE}" ]; then
    if [[ "${VALUE}" = /* ]]; then CALIB_PATH="${VALUE}"; else CALIB_PATH="${CONFIG_DIR}/${VALUE}"; fi
    if [ ! -f "${CALIB_PATH}" ]; then
      echo "Missing ${KEY} calibration: ${CALIB_PATH}" >&2
      exit 2
    fi
    SED_ARGS+=(-e "s|^${KEY}:.*|${KEY}: \"${CALIB_PATH}\"|")
  fi
done
sed -i -E "${SED_ARGS[@]}" "${EFFECTIVE_CONFIG}"
sed -i '/^diagnostics:/d' "${EFFECTIVE_CONFIG}"
printf 'diagnostics: %s\n' "${DIAGNOSTICS}" >>"${EFFECTIVE_CONFIG}"

ESTIMATOR_PID=""
MONITOR_PID=""
TEGRA_PID=""
PLAY_PID=""
COMPLETE=0

cleanup() {
  STATUS="$?"
  trap - EXIT INT TERM
  if [ -n "${PLAY_PID}" ]; then
    kill -TERM -- "-${PLAY_PID}" 2>/dev/null || true
    wait "${PLAY_PID}" 2>/dev/null || true
  fi
  if [ -n "${ESTIMATOR_PID}" ] && kill -0 "${ESTIMATOR_PID}" 2>/dev/null; then
    kill -TERM "${ESTIMATOR_PID}" 2>/dev/null || true
    sleep 1
    kill -KILL "${ESTIMATOR_PID}" 2>/dev/null || true
  fi
  if [ -n "${MONITOR_PID}" ]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [ -n "${TEGRA_PID}" ]; then
    kill "${TEGRA_PID}" 2>/dev/null || true
  fi
  if [ "${STATUS}" -eq 0 ] && [ "${COMPLETE}" -eq 1 ]; then
    printf 'SUCCESS\n' >"${RUN_DIR}/status.txt"
  else
    printf 'FAILED exit=%s\n' "${STATUS}" >"${RUN_DIR}/status.txt"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [ -n "${FW_VINS_SETUP:-}" ]; then
  source "${FW_VINS_SETUP}"
elif [ -f "${ROOT}/../../install/local_setup.bash" ]; then
  source /opt/ros/humble/setup.bash
  source "${ROOT}/../../install/local_setup.bash"
elif [ -f "${ROOT}/../install/local_setup.bash" ]; then
  source /opt/ros/humble/setup.bash
  source "${ROOT}/../install/local_setup.bash"
elif [ -f "${ROOT}/install/local_setup.bash" ]; then
  source /opt/ros/humble/setup.bash
  source "${ROOT}/install/local_setup.bash"
else
  echo "Source the ROS workspace or set FW_VINS_SETUP to its setup script" >&2
  exit 2
fi
export ROS_DOMAIN_ID="${VINS_BENCHMARK_DOMAIN_ID:-47}"

STARTED_AT="$(date --iso-8601=seconds)"
START_EPOCH="$(date +%s)"
if [ -f "${ROOT}/../../install/vins/lib/vins/vins_node" ]; then
  DEFAULT_VINS_BINARY="${ROOT}/../../install/vins/lib/vins/vins_node"
elif [ -f "${ROOT}/install/vins/lib/vins/vins_node" ]; then
  DEFAULT_VINS_BINARY="${ROOT}/install/vins/lib/vins/vins_node"
else
  DEFAULT_VINS_BINARY="${ROOT}/../install/vins/lib/vins/vins_node"
fi
VINS_NODE_BINARY="${VINS_NODE_BINARY:-${DEFAULT_VINS_BINARY}}"
stdbuf -oL -eL "${VINS_NODE_BINARY}" "${EFFECTIVE_CONFIG}" \
  >"${RUN_DIR}/vins.log" 2>&1 &
ESTIMATOR_PID="$!"

(
  echo "elapsed_s,estimator_cpu_percent,estimator_rss_kb"
  while kill -0 "${ESTIMATOR_PID}" 2>/dev/null; do
    NOW="$(date +%s)"
    read -r CPU RSS < <(ps -p "${ESTIMATOR_PID}" -o %cpu= -o rss= 2>/dev/null || true)
    if [ -n "${CPU:-}" ] && [ -n "${RSS:-}" ]; then
      echo "$((NOW - START_EPOCH)),${CPU},${RSS}"
    fi
    sleep 1
  done
) >"${RUN_DIR}/resource_samples.csv" &
MONITOR_PID="$!"

if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 1000 >"${RUN_DIR}/tegrastats.log" 2>&1 &
  TEGRA_PID="$!"
else
  : >"${RUN_DIR}/tegrastats.log"
fi

sleep 3
if ! kill -0 "${ESTIMATOR_PID}" 2>/dev/null; then
  echo "Estimator exited during startup; inspect ${RUN_DIR}/vins.log" >&2
  exit 1
fi

TOPICS=("${IMU_TOPIC}" "/camera/infra1/image_rect_raw")
if [ "${MODE}" = "stereo" ]; then
  TOPICS+=("/camera/infra2/image_rect_raw")
fi

PLAYBACK_START="$(date +%s)"
setsid ros2 bag play "${BAG}" --rate 1.0 --disable-keyboard-controls --topics "${TOPICS[@]}" \
  >"${RUN_DIR}/bag.log" 2>&1 &
PLAY_PID="$!"
BAG_STATUS=0
wait "${PLAY_PID}" || BAG_STATUS="$?"
# A bag player may leave publishers behind after an abnormal exit.
kill -TERM -- "-${PLAY_PID}" 2>/dev/null || true
PLAY_PID=""
PLAYBACK_END="$(date +%s)"
PLAYBACK_WALL="$((PLAYBACK_END - PLAYBACK_START))"

ROWS_AT_PLAYBACK_END=0
if [ -f "${RUN_DIR}/vio.csv" ]; then
  ROWS_AT_PLAYBACK_END="$(wc -l < "${RUN_DIR}/vio.csv")"
fi

ESTIMATOR_ALIVE_AFTER_PLAYBACK=0
if kill -0 "${ESTIMATOR_PID}" 2>/dev/null; then
  ESTIMATOR_ALIVE_AFTER_PLAYBACK=1
  sleep 5
fi

ROWS_AFTER_DRAIN=0
if [ -f "${RUN_DIR}/vio.csv" ]; then
  ROWS_AFTER_DRAIN="$(wc -l < "${RUN_DIR}/vio.csv")"
fi

if kill -0 "${ESTIMATOR_PID}" 2>/dev/null; then
  kill -INT "${ESTIMATOR_PID}" 2>/dev/null || true
  sleep 2
fi
if kill -0 "${ESTIMATOR_PID}" 2>/dev/null; then
  kill -TERM "${ESTIMATOR_PID}" 2>/dev/null || true
  sleep 1
fi
if kill -0 "${ESTIMATOR_PID}" 2>/dev/null; then
  kill -KILL "${ESTIMATOR_PID}" 2>/dev/null || true
fi
set +e
wait "${ESTIMATOR_PID}" 2>/dev/null
ESTIMATOR_STATUS="$?"
set -e
ESTIMATOR_PID=""

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
MONITOR_PID=""
kill "${TEGRA_PID}" 2>/dev/null || true
wait "${TEGRA_PID}" 2>/dev/null || true
TEGRA_PID=""

CPU_AVG="$(awk -F, 'NR > 1 {sum += $2; count++} END {if (count) printf "%.3f", sum/count; else print "nan"}' "${RUN_DIR}/resource_samples.csv")"
RSS_AVG="$(awk -F, 'NR > 1 {sum += $3; count++} END {if (count) printf "%.0f", sum/count; else print "nan"}' "${RUN_DIR}/resource_samples.csv")"
RSS_MAX="$(awk -F, 'NR > 1 {if ($3 > max) max=$3} END {if (max) printf "%.0f", max; else print "nan"}' "${RUN_DIR}/resource_samples.csv")"
GPU_AVG="$(awk '{for (i=1; i<=NF; i++) if ($i == "GR3D_FREQ") {value=$(i+1); gsub(/%/, "", value); sum+=value; count++}} END {if (count) printf "%.3f", sum/count; else print "nan"}' "${RUN_DIR}/tegrastats.log")"
SOLVER_AVG="$(awk '/solver costs:/ {for (i=1; i<=NF; i++) if ($i == "costs:") {sum += $(i+1); count++}} END {if (count) printf "%.6f", sum/count; else print "nan"}' "${RUN_DIR}/vins.log")"
SOLVER_MAX="$(awk '/solver costs:/ {for (i=1; i<=NF; i++) if ($i == "costs:" && $(i+1) > max) max=$(i+1)} END {if (max) printf "%.6f", max; else print "nan"}' "${RUN_DIR}/vins.log")"

TRAJECTORY_SPAN="nan"
if [ -s "${RUN_DIR}/vio.csv" ]; then
  TRAJECTORY_SPAN="$(awk -F, 'NR == 1 {first=$1} {last=$1} END {printf "%.6f", last-first}' "${RUN_DIR}/vio.csv")"
fi

cat >"${RUN_DIR}/summary.csv" <<EOF
metric,value,unit
bag_status,${BAG_STATUS},exit_code
estimator_status,${ESTIMATOR_STATUS},exit_code
estimator_alive_after_playback,${ESTIMATOR_ALIVE_AFTER_PLAYBACK},boolean
playback_wall,${PLAYBACK_WALL},seconds
trajectory_rows,${ROWS_AFTER_DRAIN},count
trajectory_rows_at_playback_end,${ROWS_AT_PLAYBACK_END},count
trajectory_span,${TRAJECTORY_SPAN},seconds
solver_average,${SOLVER_AVG},ms
solver_maximum,${SOLVER_MAX},ms
estimator_cpu_average,${CPU_AVG},percent
estimator_rss_average,${RSS_AVG},KB
estimator_rss_maximum,${RSS_MAX},KB
gpu_utilization_average,${GPU_AVG},percent
EOF

cat >"${RUN_DIR}/run_info.txt" <<EOF
bag=${BAG}
mode=${MODE}
compute=${COMPUTE}
profile=${PROFILE}
rate=1.0
rviz=0
use_gpu=${USE_GPU}
use_gpu_acc_flow=${USE_GPU_FLOW}
use_gpu_ceres=${USE_GPU_CERES}
diagnostics=${DIAGNOSTICS}
started_at=${STARTED_AT}
finished_at=$(date --iso-8601=seconds)
run_dir=${RUN_DIR}
EOF

if [ "${BAG_STATUS}" -ne 0 ] || [ "${ESTIMATOR_STATUS}" -ne 0 ] || [ "${ESTIMATOR_ALIVE_AFTER_PLAYBACK}" -ne 1 ] || [ "${ROWS_AFTER_DRAIN}" -le 0 ]; then
  echo "Run failed (bag=${BAG_STATUS}, estimator=${ESTIMATOR_STATUS}, estimator_alive=${ESTIMATOR_ALIVE_AFTER_PLAYBACK}, rows=${ROWS_AFTER_DRAIN}): ${RUN_DIR}" >&2
  exit 1
fi
COMPLETE=1
echo "${RUN_DIR}"
