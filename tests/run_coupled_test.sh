#!/usr/bin/env bash

set -u

if [ "$#" -lt 9 ]; then
  echo "Usage: $0 KIND WORK_DIR RESULT REFERENCE NUMDIFF TOLERANCE -- COMMAND --and COMMAND" >&2
  exit 2
fi

result_kind=$1
work_directory=$2
result_file=$3
reference_file=$4
numdiff_executable=$5
tolerance=$6
shift 6

if [ "$1" != "--" ]; then
  echo "Missing command separator '--'." >&2
  exit 2
fi
shift

participant_one=()
participant_two=()
active_command=one
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--and" ]; then
    if [ "$active_command" = two ]; then
      echo "Only two participant commands are supported." >&2
      exit 2
    fi
    active_command=two
  elif [ "$active_command" = one ]; then
    participant_one+=("$1")
  else
    participant_two+=("$1")
  fi
  shift
done

if [ "${#participant_one[@]}" -eq 0 ] ||
   [ "${#participant_two[@]}" -eq 0 ]; then
  echo "Both participant commands must be provided." >&2
  exit 2
fi

cd "${work_directory}" || exit 2

# Match the oversubscription and non-binding behavior used by deal.II's test
# runner. Open MPI 4 and PRRTE/Open MPI 5 use different variable names.
export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
export OMPI_MCA_hwloc_base_binding_policy=none
export PRTE_MCA_hwloc_default_binding_policy=none

participant_one_log=participant-one.log
participant_two_log=participant-two.log
children=()

terminate_children()
{
  if [ "${#children[@]}" -gt 0 ]; then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
    children=()
  fi
}

show_participant_logs()
{
  echo "----- participant one -----"
  cat "${participant_one_log}" 2>/dev/null || true
  echo "----- participant two -----"
  cat "${participant_two_log}" 2>/dev/null || true
}

show_comparison_failure()
{
  echo "=== REGRESSION COMPARISON FAILED ==="
  echo "Generated: ${work_directory}/${result_file}"
  echo "Reference: ${reference_file}"
  echo "numdiff legend: '<==' is generated, '==>' is reference."
  echo
  cat comparison.diff

  mapfile -t differing_lines < <(
    sed -n 's/^##\([0-9][0-9]*\).*$/\1/p' comparison.diff | sort -nu
  )

  if [ "${#differing_lines[@]}" -gt 0 ]; then
    echo "----- marked differing lines -----"
    for line_number in "${differing_lines[@]}"; do
      generated_line=$(sed -n "${line_number}p" "${result_file}")
      reference_line=$(sed -n "${line_number}p" "${reference_file}")
      printf '>>> GENERATED line %s: %s\n' \
        "${line_number}" "${generated_line}"
      printf '>>> REFERENCE line %s: %s\n' \
        "${line_number}" "${reference_line}"
    done
  else
    echo "----- structural diff -----"
    diff -u "${reference_file}" "${result_file}" | sed -n '1,120p' || true
  fi

  echo "Participant logs were successful and remain available at:"
  echo "  ${work_directory}/${participant_one_log}"
  echo "  ${work_directory}/${participant_two_log}"
}

trap terminate_children EXIT
trap 'exit 130' INT TERM

"${participant_one[@]}" > "${participant_one_log}" 2>&1 &
children+=("$!")
"${participant_two[@]}" > "${participant_two_log}" 2>&1 &
children+=("$!")

# Return promptly if either participant fails instead of waiting for CTest's
# timeout while its coupling partner blocks.
wait -n
first_result=$?
if [ "${first_result}" -ne 0 ]; then
  terminate_children
  echo "A coupled participant failed with exit code ${first_result}." >&2
  show_participant_logs
  exit "${first_result}"
fi

wait -n
second_result=$?
children=()
if [ "${second_result}" -ne 0 ]; then
  echo "A coupled participant failed with exit code ${second_result}." >&2
  show_participant_logs
  exit "${second_result}"
fi

case "${result_kind}" in
  existing)
    ;;
  solid)
    ;;
  extract-errors)
    if ! grep -F "Error" "${participant_one_log}" > "${result_file}"; then
      echo "Participant one produced no error measurements." >&2
      show_participant_logs
      exit 1
    fi
    if ! grep -F "Error" "${participant_two_log}" >> "${result_file}"; then
      echo "Participant two produced no error measurements." >&2
      show_participant_logs
      exit 1
    fi
    ;;
  *)
    echo "Unknown result kind: ${result_kind}" >&2
    exit 2
    ;;
esac

if [ ! -f "${result_file}" ]; then
  echo "Expected result file was not generated: ${result_file}" >&2
  show_participant_logs
  exit 1
fi

numdiff_arguments=()
comparison_result_file=${result_file}
comparison_reference_file=${reference_file}

if [ "${result_kind}" = solid ]; then
  cg_relative_tolerance=${tolerance}
  generated_cg_count=$(
    sed -n 's/^Average CG iter = \([0-9][0-9]*\)$/\1/p' "${result_file}"
  )
  reference_cg_count=$(
    sed -n 's/^Average CG iter = \([0-9][0-9]*\)$/\1/p' "${reference_file}"
  )
  generated_cg_line=$(
    grep -n -m 1 '^Average CG iter = ' "${result_file}" | cut -d: -f1
  )

  if [ -z "${generated_cg_count}" ] ||
     [ -z "${reference_cg_count}" ] ||
     [ -z "${generated_cg_line}" ]; then
    echo "Could not read one integer 'Average CG iter' value from both files." \
      > comparison.diff
    show_comparison_failure
    exit 1
  fi

  comparison_result_file=result.with-normalized-cg
  comparison_reference_file=reference.with-normalized-cg
  sed 's/^Average CG iter = .*/Average CG iter = <checked separately>/' \
    "${result_file}" > "${comparison_result_file}"
  sed 's/^Average CG iter = .*/Average CG iter = <checked separately>/' \
    "${reference_file}" > "${comparison_reference_file}"
elif [ "${tolerance}" != exact ]; then
  numdiff_arguments+=("--absolute-tolerance=${tolerance}")
fi

if ! "${numdiff_executable}" "${numdiff_arguments[@]}" \
    "${comparison_result_file}" "${comparison_reference_file}" \
    > comparison.diff 2>&1; then
  show_comparison_failure
  exit 1
fi

if [ "${result_kind}" = solid ] &&
   ! awk -v generated="${generated_cg_count}" \
         -v reference="${reference_cg_count}" \
         -v relative_tolerance="${cg_relative_tolerance}" \
         'BEGIN {
            allowance = reference * relative_tolerance
            exit !(generated > 0 &&
                   reference > 0 &&
                   generated >= reference - allowance &&
                   generated <= reference + allowance)
          }'; then
  {
    echo "----------------"
    printf '##%s      #:5   <== %s\n' \
      "${generated_cg_line}" "${generated_cg_count}"
    printf '##%s      #:5   ==> %s\n' \
      "${generated_cg_line}" "${reference_cg_count}"
    printf '@ CG iteration count must be positive and within +/-%g%% of the reference\n' \
      "$(awk -v tolerance="${cg_relative_tolerance}" \
             'BEGIN { print 100 * tolerance }')"
    printf '@ Accepted range = [%.6g, %.6g]\n' \
      "$(awk -v reference="${reference_cg_count}" \
             -v relative_tolerance="${cg_relative_tolerance}" \
             'BEGIN { print reference * (1 - relative_tolerance) }')" \
      "$(awk -v reference="${reference_cg_count}" \
             -v relative_tolerance="${cg_relative_tolerance}" \
             'BEGIN { print reference * (1 + relative_tolerance) }')"
  } > comparison.diff
  show_comparison_failure
  exit 1
fi

rm -f result.with-normalized-cg reference.with-normalized-cg
: > comparison.diff
