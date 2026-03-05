#!/bin/bash
# filepath: /home/jh2589/placebo_test/tmf-implementation/pipeline.sh

# Determine script dir early so we can locate config files relative to the script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try several candidate locations for the config (script dir, script dir/conf, current dir)
CFG_CANDIDATES=(
  "${SCRIPT_DIR}/pipeline.conf"
  "${SCRIPT_DIR}/conf/pipeline.conf"
  "./pipeline.conf"
)

CFG=""
for c in "${CFG_CANDIDATES[@]}"; do
  if [ -f "$c" ]; then
    CFG="$c"
    break
  fi
done

if [ -z "$CFG" ]; then
  echo "ERROR: config file not found. Looked for:"
  for c in "${CFG_CANDIDATES[@]}"; do echo "  - $c"; done
  exit 1
fi

# load config
source "$CFG"

# Optional failure notifier hook. If not provided by config/sourced scripts,
# keep pipeline running without crashing.
if ! declare -F send_email_on_failure >/dev/null 2>&1; then
  function send_email_on_failure {
    local proj="$1"
    local t0="$2"
    local eval_year="$3"
    echo "Warning: send_email_on_failure is not configured (project=${proj}, start=${t0}, eval=${eval_year})"
    return 0
  }
fi

# Defaults (can be overridden in pipeline.conf)
BUFFER_METERS="${BUFFER_METERS:-0}"
K_NUM_TO_KEEP="${K_NUM_TO_KEEP:-256}"
PROCESSES="${PROCESSES:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Define previous settings file in same directory as script
PREV_SETTINGS="${SCRIPT_DIR}/python_settings"

declare -A STEP_DESC=(
  [1]="Create output folder"
  [2]="Generate buffer (boundary)"
  [3]="Generate country list"
  [4]="Generate matching area"
  [5]="Download SRTM data"
  [6]="Generate slopes"
  [7]="Rescale elevation tiles"
  [8]="Rescale slope tiles"
  [9]="Generate country raster"
  [10]="Calculate set K"
  [11]="Find potential matches"
  [12]="Build M table"
  [13]="Find pairs"
  [14]="Calculate additionality"
  [15]="Generate effect raster"
)

# Add title ASCII art to show at menus
TITLE=$(cat <<'EOF'

██████╗  █████╗  ██████╗████████╗    ██╗   ██╗   ██████╗ 
██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██║   ██║   ╚════██╗
██████╔╝███████║██║        ██║       ██║   ██║    █████╔╝
██╔═══╝ ██╔══██║██║        ██║       ╚██╗ ██╔╝    ╚═══██╗
██║     ██║  ██║╚██████╗   ██║        ╚████╔╝ ██╗██████╔╝
╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝         ╚═══╝  ╚═╝╚═════╝
EOF
)

# Global variables for step selection
STEPS_TO_RUN=()
RUN_ALL=false
STEPS_RAW=""


# Function to save settings for repeat
function save_settings() {
  local mode="$1"
  local steps="$2"
  local proj="$3"
  local t0="$4"
  local eval_year="$5"
  local csv_file="$6"
  
  cat > "$PREV_SETTINGS" << EOF
MODE=$mode
STEPS_RAW=$steps
PROJ=$proj
T0=$t0
EVAL_YEAR=$eval_year
CSV_FILE=$csv_file
EOF
  echo "Settings saved for next run."
}

# Function to load previous settings
function load_previous_settings() {
  if [ -f "$PREV_SETTINGS" ]; then
    source "$PREV_SETTINGS"
    echo "Previous settings loaded:"
    if [ "$MODE" = "single" ]; then
      echo "  Mode: Single project"
      echo "  Project: $PROJ"
      echo "  Start year: $T0"
      echo "  Evaluation year: $EVAL_YEAR"
    elif [ "$MODE" = "csv" ]; then
      echo "  Mode: CSV of projects"
      echo "  CSV file: $CSV_FILE"
    fi
    echo "  Steps: $STEPS_RAW"
    return 0
  else
    echo "No previous settings found."
    return 1
  fi
}

function select_steps {
  STEPS_TO_RUN=()
  RUN_ALL=false
  STEPS_RAW=""

  # Use arrow-key menu to choose All or Specify
  if ! terminal_menu "Which steps would you like to run? (Enter to select)" "All steps" "Specify steps" "Cancel"; then
    echo "Steps selection canceled."
    return 1
  fi

  case "$REPLY" in
    1)
      RUN_ALL=true
      STEPS_RAW="all"
      ;;
    2)
      # Show available steps before asking for input
      echo ""
      echo "Available steps:"
      for k in $(printf "%s\n" "${!STEP_DESC[@]}" | sort -n); do
        printf "  %2s) %s\n" "$k" "${STEP_DESC[$k]}"
      done
      echo ""
      # text input for specifying steps (e.g. 3-6,8,10)
      read -p "Enter steps (e.g. 3-6,8,10): " STEPS_RAW
      if [ -z "$STEPS_RAW" ]; then
        echo "No steps specified. Canceling."
        return 1
      fi
      STEPS_TO_RUN=()
      IFS=',' read -ra TOKENS <<< "$STEPS_RAW"
      for t in "${TOKENS[@]}"; do
        if [[ $t =~ ^([0-9]+)-([0-9]+)$ ]]; then
          start=${BASH_REMATCH[1]}
          end=${BASH_REMATCH[2]}
          for ((n=start; n<=end; n++)); do
            STEPS_TO_RUN+=("$n")
          done
        elif [[ $t =~ ^[0-9]+$ ]]; then
          STEPS_TO_RUN+=("$t")
        else
          echo "Warning: invalid token '$t' skipped"
        fi
      done
      RUN_ALL=false
      ;;
    3)
      echo "Steps selection canceled."
      return 1
      ;;
  esac
}

function should_run {
  local S=$1
  # explicit check for RUN_ALL=true
  if [ "$RUN_ALL" = true ]; then
    return 0
  fi
  for x in "${STEPS_TO_RUN[@]}"; do
    [ "$x" -eq "$S" ] && return 0
  done
  return 1
}

function run_single_project {
  read -p "Enter project name (ID): " proj
  read -p "Enter start year: " t0
  read -p "Enter evaluation year: " eval_year
  
  # Save settings immediately after gathering them
  save_settings "single" "$STEPS_RAW" "$proj" "$t0" "$eval_year" ""
  
  echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
  if ! run_pipeline "$proj" "$t0" "$eval_year"; then
    # keep same failure handling as repeat path
    send_email_on_failure "$proj" "$t0" "$eval_year"
  fi
}

function run_csv_projects {
  read -p "Enter CSV file to use [project_metadata.csv]: " CSV_FILE
  CSV_FILE=${CSV_FILE:-project_metadata.csv}
  if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file $CSV_FILE not found."
    return
  fi
  
  # Save settings immediately
  save_settings "csv" "$STEPS_RAW" "" "" "" "$CSV_FILE"
  
  # Read CSV lines into an array to avoid subshell issues
  mapfile -t lines < <(tail -n +2 "$CSV_FILE")
  
  for line in "${lines[@]}"; do
    IFS=',' read -r proj t0 eval_year <<< "$line"
    echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
    run_pipeline "$proj" "$t0" "$eval_year"
  done
}

# new helper: run a command and append both stdout+stderr with a timestamped header to per-project terminal log
function run_and_log {
  local proj="$1"; shift
  local log_file="${OUTPUT_DIR}/${proj}/terminal.log"
  mkdir -p "$(dirname "$log_file")"
  printf "\n[%s] RUN: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$log_file"
  # run command, stream stdout+stderr to both terminal and log; preserve command exit status
  "$@" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    printf "[%s] ERROR (rc=%d): command failed: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$rc" "$*" >> "$log_file"
  fi
  return $rc
}

function run_pipeline {
  local proj="$1"
  local t0="$2"
  local eval_year="$3"
  local FAILED=0
  local LOG_DIR="${OUTPUT_DIR}/${proj}"
  local TERMINAL_LOG="${LOG_DIR}/terminal.log"

  if should_run 1; then
    mkdir -p "${OUTPUT_DIR}/${proj}"
    : > "$TERMINAL_LOG"  # create/truncate per-project terminal log
    echo "--Folder created.--"
  fi

  if should_run 2; then
    if run_and_log "$proj" python3 -m methods.inputs.leakage_buffer \
      --project "${INPUT_DIR}/${proj}.geojson" \
      --out "${OUTPUT_DIR}/${proj}/buffered_project.geojson" \
      --buffer "${BUFFER_METERS}" \
      --countries "${COUNTRIES_RASTER}"; then
      echo "--Buffer expansion created.--"
    else
      echo "ERROR: leakage_buffer failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 3; then
    if run_and_log "$proj" python3 -m methods.inputs.generate_country_list \
      --project "${OUTPUT_DIR}/${proj}/buffered_project.geojson" \
      --countries "${COUNTRIES_RASTER}" \
      --output "${OUTPUT_DIR}/${proj}/country-list.json"; then
      echo "--Country list created.--"
    else
      echo "ERROR: generate_country_list failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 4; then
    if run_and_log "$proj" python3 -m methods.inputs.generate_matching_area \
      --matching_radius_metres 1000000 \
      --project "${OUTPUT_DIR}/${proj}/buffered_project.geojson" \
      --countrycodes "${OUTPUT_DIR}/${proj}/country-list.json" \
      --countries "${COUNTRIES_RASTER}" \
      --ecoregions "${ECOR_GEOJSON}" \
      --projects "${OTHER_PROJECTS_DIR}" \
      --output "${OUTPUT_DIR}/${proj}/matching-area_1000.geojson"; then
      echo "--Matching area 1000km created.--"
    else
      echo "ERROR: generate_matching_area failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 5; then
    if run_and_log "$proj" python3 -m methods.inputs.download_srtm_data \
      --project "${OUTPUT_DIR}/${proj}/buffered_project.geojson" \
      --matching "${OUTPUT_DIR}/${proj}/matching-area_1000.geojson" \
      --zips "${SRTM_ZIP_DIR}" \
      --tifs "${SRTM_TIF_DIR}"; then
      echo "--SRTM downloaded.--"
    else
      echo "ERROR: download_srtm_data failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 6; then
    if run_and_log "$proj" python3 -m methods.inputs.generate_slope \
      --input "${SRTM_TIF_DIR}" \
      --output "${SRTM_TIF_DIR}/slopes"; then
      echo "--Slope created.--"
    else
      echo "ERROR: generate_slope failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 7; then
    if run_and_log "$proj" python3 -m methods.inputs.rescale_tiles_to_jrc \
      --jrc "${JRC_DIR}" \
      --tiles "${SRTM_TIF_DIR}" \
      --output "${SRTM_TIF_DIR}/rescaled-elevation"; then
      echo "--Elevation rescaled.--"
    else
      echo "ERROR: rescale elevation failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 8; then
    if run_and_log "$proj" python3 -m methods.inputs.rescale_tiles_to_jrc \
      --jrc "${JRC_DIR}" \
      --tiles "${SRTM_TIF_DIR}/slopes" \
      --output "${SRTM_TIF_DIR}/rescaled-slopes"; then
      echo "--Slopes rescaled.--"
    else
      echo "ERROR: rescale slopes failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 9; then
    if run_and_log "$proj" python3 -m methods.inputs.generate_country_raster \
      --jrc "${JRC_DIR}" \
      --matching "${OUTPUT_DIR}/${proj}/matching-area_1000.geojson" \
      --countries "${COUNTRIES_RASTER}" \
      --output "${OUTPUT_DIR}/${proj}/countries.tif"; then
      echo "--Country raster created.--"
    else
      echo "ERROR: generate_country_raster failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 10; then
    if run_and_log "$proj" python3 -m methods.matching.calculate_k \
      --project "${OUTPUT_DIR}/${proj}/buffered_project.geojson" \
      --start_year "$t0" \
      --evaluation_year "$eval_year" \
      --jrc "${JRC_DIR}" \
      --fcc "${FCC_DIR}" \
      --ecoregions "${ECOR_DIR}" \
      --elevation "${SRTM_TIF_DIR}/rescaled-elevation" \
      --slope "${SRTM_TIF_DIR}/rescaled-slopes" \
      --access "${ACCESS_DIR}" \
      --countries-raster "${OUTPUT_DIR}/${proj}/countries.tif" \
      --output "${OUTPUT_DIR}/${proj}/k_grids" \
      --num-to-keep "${K_NUM_TO_KEEP}" \
      --processes 32; then
      echo "--Set K created.--"
    else
      echo "ERROR: calculate_k failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 11; then
    if run_and_log "$proj" tmfpython3 -m methods.matching.find_potential_matches \
      --k "${OUTPUT_DIR}/${proj}/k_grids" \
      --matching "${OUTPUT_DIR}/${proj}/matching-area_1000.geojson" \
      --start_year "$t0" \
      --evaluation_year "$eval_year" \
      --jrc "${JRC_DIR}" \
      --fcc "${FCC_DIR}" \
      --ecoregions "${ECOR_DIR}" \
      --elevation "${SRTM_TIF_DIR}/rescaled-elevation" \
      --slope "${SRTM_TIF_DIR}/rescaled-slopes" \
      --access "${ACCESS_DIR}" \
      --countries-raster "${OUTPUT_DIR}/${proj}/countries.tif" \
      --j "${PROCESSES}" \
      --output "${OUTPUT_DIR}/${proj}/matches"; then
      echo "--M rasters created.--"
    else
      echo "ERROR: find_potential_matches failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 12; then
    if run_and_log "$proj" python3 -m methods.matching.build_m_table \
      --rasters_directory "${OUTPUT_DIR}/${proj}/matches/" \
      --matching "${OUTPUT_DIR}/${proj}/matching-area_1000.geojson" \
      --start_year "$t0" \
      --evaluation_year "$eval_year" \
      --jrc "${JRC_DIR}" \
      --fcc "${FCC_DIR}" \
      --ecoregions "${ECOR_DIR}" \
      --elevation "${SRTM_TIF_DIR}/rescaled-elevation" \
      --slope "${SRTM_TIF_DIR}/rescaled-slopes" \
      --access "${ACCESS_DIR}" \
      --countries-raster "${OUTPUT_DIR}/${proj}/countries.tif" \
      --output "${OUTPUT_DIR}/${proj}/matches.parquet"; then
      echo "--M table created.--"
    else
      echo "ERROR: build_m_table failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 13; then
    if run_and_log "$proj" julia --project="${SCRIPT_DIR}" "${SCRIPT_DIR}/methods/matching/find_pairs_prop_hybrid.jl" \
      --k_directory "${OUTPUT_DIR}/${proj}/k_grids" \
      --m_parquet_filename "${OUTPUT_DIR}/${proj}/matches.parquet" \
      --m_sample_folder "${OUTPUT_DIR}/${proj}/pairs_1000/m_samples/" \
      --start_year "$t0" \
      --evaluation_year "$eval_year" \
      --output_folder "${OUTPUT_DIR}/${proj}/pairs_hybrid" \
      --batch_size "${BATCH_SIZE}" \
      --processes_count "${PROCESSES}" \
      --seed 42 \
      --use-fork \
      --min_potential_matches 100 \
      --similarity_mode hybrid_mahal_prop \
      --propensity_caliper 0.05 \
      --mahalanobis_lambda 0.8 \
      --mahalanobis_bandwidth 0.25 \
      --cluster_k_first 1 \
      --max_potential_matches 500; then
      echo "--Pairs Hybrid Mahalanobis+Propensity (Julia) created.--"
    else
      echo "ERROR: find_pairs_hybrid (Julia) failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 14; then
    if run_and_log "$proj" tmfpython3 -m methods.outputs.calculate_additionality \
      --project-folder "${OUTPUT_DIR}/${proj}" \
      --pairs-folder "pairs_hybrid" \
      --og-geojson "${INPUT_DIR}/1201.geojson"; then
      echo "--Additionality calculated.--"
    else
      echo "ERROR: calculate_additionality failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  if should_run 15; then
    if run_and_log "$proj" tmfpython3 -m methods.outputs.effect_raster \
      --project-folder "${OUTPUT_DIR}/${proj}" \
      --all-pairs "${OUTPUT_DIR}/${proj}/all_pairs.parquet" \
      --countries-tif "${OUTPUT_DIR}/${proj}/countries.tif" \
      --buffered-geojson "${OUTPUT_DIR}/${proj}/buffered_project.geojson" \
      --output-parquet "${OUTPUT_DIR}/${proj}/effect.parquet" \
      --output-raster "${OUTPUT_DIR}/${proj}/effect.tif"; then
      echo "--Effect raster created.--"
    else
      echo "ERROR: effect_raster failed (logged to ${TERMINAL_LOG})"
      FAILED=1
    fi
  fi

  return $FAILED
}

# --- Terminal UI helpers ---
function terminal_select_steps {
  # Presents STEP_DESC list navigable with Up/Down, toggle with Space, 'a' toggles all, Enter=confirm, 'q' cancels.
  local keys=( $(printf "%s\n" "${!STEP_DESC[@]}" | sort -n) )
  local total=${#keys[@]}
  local cursor=0
  local offset=0
  local term_lines=$(tput lines)
  local view_height=$(( term_lines - 6 ))
  [ $view_height -lt 5 ] && view_height=5

  declare -A selected
  for k in "${keys[@]}"; do selected[$k]=false; done
  RUN_ALL=false
  STEPS_RAW=""

  tput civis
  while true; do
    tput clear
    echo "Use ↑/↓ to navigate, Space to toggle, a=toggle all, Enter=confirm, q=cancel"
    echo ""
    local start=$offset
    local end=$(( offset + view_height - 1 ))
    [ $end -gt $(( total - 1 )) ] && end=$(( total - 1 ))
    for i in $(seq $start $end); do
      k=${keys[i]}
      marker="[ ]"
      $([ "${selected[$k]}" = true ] && printf -v marker "[x]")
      if [ $i -eq $cursor ]; then
        # focused (cursor): show palm-tree emoji on each side (no ANSI highlight)
        printf "  🌴 %2s) %s %s 🌴\n" "$k" "$marker" "${STEP_DESC[$k]}"
      else
        # non-focused: plain line (still show toggle marker)
        printf "    %2s) %s %s\n" "$k" "$marker" "${STEP_DESC[$k]}"
      fi
    done
    # read up to 3 chars for arrows
    IFS= read -rsn3 key
    case "$key" in
      $'\e[A') # up
        if [ $cursor -gt 0 ]; then cursor=$((cursor-1)); fi
        if [ $cursor -lt $offset ]; then offset=$cursor; fi
        ;;
      $'\e[B') # down
        if [ $cursor -lt $(( total - 1 )) ]; then cursor=$((cursor+1)); fi
        if [ $cursor -gt $end ]; then offset=$((offset+1)); fi
        ;;
      ' ') # toggle
        k=${keys[cursor]}
        if [ "${selected[$k]}" = true ]; then selected[$k]=false; else selected[$k]=true; fi
        ;;
      a)
        # toggle all
        local any=false
        for k in "${keys[@]}"; do
          $([ "${selected[$k]}" = true ] && any=true)
        done
        if $any; then
          for k in "${keys[@]}"; do selected[$k]=false; done
        else
          for k in "${keys[@]}"; do selected[$k]=true; done
        fi
        ;;
      '') # Enter
        # build STEPS_TO_RUN / RUN_ALL / STEPS_RAW
        STEPS_TO_RUN=()
        local sr=""
        for k in "${keys[@]}"; do
          if [ "${selected[$k]}" = true ]; then
            STEPS_TO_RUN+=("$k")
            sr="${sr}${k},"
          fi
        done
        if [ ${#STEPS_TO_RUN[@]} -eq 0 ]; then
          RUN_ALL=true
          STEPS_RAW="all"
        else
          RUN_ALL=false
          # remove trailing comma
          STEPS_RAW="${sr%,}"
        fi
        tput cnorm
        return 0
        ;;
      q)
        tput cnorm
        return 1
        ;;
    esac
  done
}

function terminal_menu {
  local prompt="$1"; shift
  local options=("$@")
  local count=${#options[@]}
  local cursor=0

  tput civis
  while true; do
    tput clear
    # print TITLE in dark green
    printf "\e[32m%s\e[0m\n" "$TITLE"
    echo ""
    echo "$prompt"
    echo ""
    # focused option: palm-tree emoji markers; others plain
    for i in "${!options[@]}"; do
      if [ "$i" -eq "$cursor" ]; then
        printf "  🌴 %s🌴\n" "${options[i]}"
      else
        printf "    %s\n" "${options[i]}"
      fi
    done

    # read keys: handle arrows (ESC [ A/B), Enter and 'q'
    local key=""
    local k1 k2 k3
    IFS= read -rsn1 k1
    if [ -z "$k1" ]; then
      key=$'\n'
    elif [ "$k1" = $'\e' ]; then
      IFS= read -rsn1 -t 0.01 k2 || k2=""
      IFS= read -rsn1 -t 0.01 k3 || k3=""
      key=$'\e'"$k2$k3"
    else
      key="$k1"
    fi

    case "$key" in
      $'\e[A') [ $cursor -gt 0 ] && cursor=$((cursor-1)) ;;   # up
      $'\e[B') [ $cursor -lt $((count-1)) ] && cursor=$((cursor+1)) ;; # down
      $'\n') tput cnorm; REPLY=$((cursor+1)); return 0 ;; # Enter
      q) tput cnorm; return 1 ;; # quit
      *) ;; # ignore other input
    esac
  done
}

function terminal_run_ui {
  # choose mode using arrow-key menu (replaces select/PS3)
  if ! terminal_menu "Terminal UI: choose mode (Enter to select)" "Single project" "CSV of projects" "Cancel"; then
    echo "Mode selection canceled."
    return
  fi
  case "$REPLY" in
    1) MODE="single" ;;
    2) MODE="csv" ;;
    3) return ;;
  esac

  # select steps via text UI (All or specify like 3-6,8)
  if ! select_steps; then
    echo "Steps selection canceled."
    return
  fi

  if [ "$MODE" = "single" ]; then
    read -p "Enter project name (ID): " proj
    read -p "Enter start year: " t0
    read -p "Enter evaluation year: " eval_year
    save_settings "single" "$STEPS_RAW" "$proj" "$t0" "$eval_year" ""
    echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
    if ! run_pipeline "$proj" "$t0" "$eval_year"; then
      send_email_on_failure "$proj" "$t0" "$eval_year"
    fi
  else
    read -p "Enter CSV file to use [project_metadata.csv]: " CSV_FILE
    CSV_FILE=${CSV_FILE:-project_metadata.csv}
    if [ ! -f "$CSV_FILE" ]; then
      echo "ERROR: CSV file $CSV_FILE not found."
      return
    fi
    save_settings "csv" "$STEPS_RAW" "" "" "" "$CSV_FILE"
    mapfile -t lines < <(tail -n +2 "$CSV_FILE")
    for line in "${lines[@]}"; do
      IFS=',' read -r proj t0 eval_year <<< "$line"
      echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
      if ! run_pipeline "$proj" "$t0" "$eval_year"; then
        send_email_on_failure "$proj" "$t0" "$eval_year"
      fi
    done
  fi
}

# open pipeline.conf in terminal editor (respects $EDITOR/$VISUAL, falls back)
function edit_settings {
  local editor="${EDITOR:-${VISUAL:-}}"
  if [ -z "$editor" ]; then
    if command -v nano >/dev/null 2>&1; then
      editor="nano"
    elif command -v vim >/dev/null 2>&1; then
      editor="vim"
    else
      editor="vi"
    fi
  fi

  echo "Opening $CFG with $editor..."
  "$editor" "$CFG"
}

# Replace the old numeric Main menu loop with an arrow-key-only UI loop
while true; do
  if ! terminal_menu "Main menu — use ↑/↓ and Enter to choose" "Single project" "CSV of projects" "Repeat prior instructions" "Settings" "Exit"; then
    echo "Selection canceled. Exiting."
    exit 0
  fi

  case "$REPLY" in
    1) # Single project
      # choose steps (all or specify via text)
      if ! select_steps; then
        continue
      fi
      read -p "Enter project name (ID): " proj
      read -p "Enter start year: " t0
      read -p "Enter evaluation year: " eval_year
      save_settings "single" "$STEPS_RAW" "$proj" "$t0" "$eval_year" ""
      echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
      if ! run_pipeline "$proj" "$t0" "$eval_year"; then
        send_email_on_failure "$proj" "$t0" "$eval_year"
      fi
      ;;
    2) # CSV of projects
      if ! select_steps; then
        continue
      fi
      read -p "Enter CSV file to use [project_metadata.csv]: " CSV_FILE
      CSV_FILE=${CSV_FILE:-project_metadata.csv}
      if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: CSV file $CSV_FILE not found."
        continue
      fi
      save_settings "csv" "$STEPS_RAW" "" "" "" "$CSV_FILE"
      mapfile -t lines < <(tail -n +2 "$CSV_FILE")
      for line in "${lines[@]}"; do
        IFS=',' read -r proj t0 eval_year <<< "$line"
        echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
        if ! run_pipeline "$proj" "$t0" "$eval_year"; then
          send_email_on_failure "$proj" "$t0" "$eval_year"
        fi
      done
      ;;
    3) # Repeat prior instructions
      if ! load_previous_settings; then
        echo "No prior settings to repeat."
        continue
      fi
      # rebuild STEPS_TO_RUN from STEPS_RAW
      if [ "$STEPS_RAW" != "" ] && [ "$STEPS_RAW" != "all" ]; then
        STEPS_TO_RUN=()
        IFS=',' read -ra TOKENS <<< "$STEPS_RAW"
        for t in "${TOKENS[@]}"; do
          if [[ $t =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            for ((n=start; n<=end; n++)); do
              STEPS_TO_RUN+=("$n")
            done
          elif [[ $t =~ ^[0-9]+$ ]]; then
            STEPS_TO_RUN+=("$t")
          fi
        done
        RUN_ALL=false
      else
        RUN_ALL=true
      fi

      if [ "$MODE" = "single" ]; then
        echo "Running pipeline for $PROJ (start: $T0, eval: $EVAL_YEAR)"
        if ! run_pipeline "$PROJ" "$T0" "$EVAL_YEAR"; then
          send_email_on_failure "$PROJ" "$T0" "$EVAL_YEAR"
        fi
      elif [ "$MODE" = "csv" ]; then
        if [ ! -f "$CSV_FILE" ]; then
          echo "ERROR: CSV file $CSV_FILE not found."
          continue
        fi
        mapfile -t lines < <(tail -n +2 "$CSV_FILE")
        for line in "${lines[@]}"; do
          IFS=',' read -r proj t0 eval_year <<< "$line"
          echo "Running pipeline for $proj (start: $t0, eval: $eval_year)"
          if ! run_pipeline "$proj" "$t0" "$eval_year"; then
            send_email_on_failure "$proj" "$t0" "$eval_year"
          fi
        done
      else
        echo "Unknown MODE in previous settings: $MODE"
      fi
      ;;
    4) # Settings
      edit_settings
      ;;
    5)
      echo "Exiting."
      exit 0
      ;;
    *)
      echo "Invalid choice."
      ;;
  esac
done