#!/bin/bash

set -u
echo "Testing the heat transfer"

RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

start_dir=$(pwd)

# Change to the working directory
cd "$(dirname "$0")" || exit 1
work_dir=$(pwd)

# Declare test names
declare -a tests=("partitioned-heat" "partitioned-heat-direct-access")

exit_code=0

print_start(){
	echo -e " ***Test $*:"
	echo -ne "\t Starting ..."
}

print_result() {
	if [ $? -eq 0 ]
	then
            echo -ne "${GREEN} passed ${NOCOLOR}\n"
	else
            echo -ne "${RED} failed ${NOCOLOR}\n"
            exit_code=$(( exit_code +1))
            cat "$@"
            echo "----- Test Result -----"
            cat "$i"/output
	fi
}

test_name="building"
print_start ${test_name}
mkdir -p "${work_dir}"/build && cd "${work_dir}"/build || exit 1
(cmake ../../../ && make debug && make heat) 2>&1 | tee "${test_name}.log"
 if [ $? -eq 0 ]
    then
    echo -ne "${GREEN} passed ${NOCOLOR}\n"
    else
    echo -ne "${RED} failed ${NOCOLOR}\n"
    exit 1
 fi

cd "${work_dir}" || exit
rm -fv ./heat
cp ./build/heat .

# Specify the precice-config
for i in "${tests[@]}"
    do
    # Serial
    test_name="${i}_serial"
    print_start "${test_name}"
    ./heat "$i"/dirichlet.prm > "$i"/dirichlet.log & ./heat "$i"/neumann.prm > "$i"/neumann.log
    timeout 120s ./heat "$i"/dirichlet.prm > "$i"/dirichlet.log 2>&1 &
    pid_dirichlet=$!
    timeout 120s ./heat "$i"/neumann.prm > "$i"/neumann.log 2>&1 &
    pid_neumann=$!

    # Wait for both processes to finish and capture exit codes
    wait $pid_dirichlet; rc_dirichlet=$?
    wait $pid_neumann; rc_neumann=$?
    if [ "$rc_dirichlet" -ne 0 ] || [ "$rc_neumann" -ne 0 ]; then
        echo "Test ${test_name} failed (dirichlet exit code: $rc_dirichlet, neumann exit code: $rc_neumann)."
        echo "----- dirichlet.log -----"
        cat "$i"/dirichlet.log
        echo "----- neumann.log -----"
        cat "$i"/neumann.log
        exit 1
    fi
    grep 'Error' "$i"/dirichlet.log > "$i"/output
    grep 'Error' "$i"/neumann.log >> "$i"/output
    numdiff --absolute-tolerance=9e-14 "$i"/output "$i"/"${test_name}".output > "$i"/"${test_name}".diff
    print_result "$i"/"${test_name}".diff

    # Parallel
    test_name="${i}_parallel"
    print_start "${test_name}"
    timeout 120s mpirun --oversubscribe -np 4 ./heat "$i"/dirichlet.prm > "$i"/dirichlet.log 2>&1 &
    pid_dirichlet=$!
    timeout 120s mpirun --oversubscribe -np 4 ./heat "$i"/neumann.prm > "$i"/neumann.log 2>&1 &
    pid_neumann=$!

    wait $pid_dirichlet; rc_dirichlet=$?
    wait $pid_neumann; rc_neumann=$?
    if [ "$rc_dirichlet" -ne 0 ] || [ "$rc_neumann" -ne 0 ]; then
        echo "Test ${test_name} failed (dirichlet exit code: $rc_dirichlet, neumann exit code: $rc_neumann)."
        echo "----- dirichlet.log -----"
        cat "$i"/dirichlet.log
        echo "----- neumann.log -----"
        cat "$i"/neumann.log
        exit 1
    fi
    grep 'Error' "$i"/dirichlet.log > "$i"/output
    grep 'Error' "$i"/neumann.log >> "$i"/output
    numdiff --absolute-tolerance=9e-14 "$i"/output "$i"/"${test_name}".output > "$i"/"${test_name}".diff
    print_result "$i"/"${test_name}".diff
done

# Go back to initial to command directory
cd "${start_dir}" || exit 1

if [ $exit_code -eq 0 ]
then
    echo "All tests passed."
else
    echo "Errors occurred: $exit_code tests failed."
    exit 1
fi
