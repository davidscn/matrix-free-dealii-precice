#!/bin/bash

set -u
echo "Testing the solid mechanics"

RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

start_dir=$(pwd)

# Change to the working directory
cd "$(dirname "$0")"
work_dir=$(pwd)

# Declare test names
declare -a tests=("turek_gmg_scalar_referential" "turek_gmg_tensor2" "turek_gmg_tensor4" "turek_gmg_tensor4_ns" "turek_gmg_tensor2_form0" "turek_jacobi_tensor2")

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
mkdir -p "${work_dir}"/build && cd "${work_dir}"/build
(cmake ../../../ && make debug && make solid) 2>&1 | tee "${test_name}.log"
 if [ $? -eq 0 ]
    then
    echo -ne "${GREEN} passed ${NOCOLOR}\n"
    else
    echo -ne "${RED} failed ${NOCOLOR}\n"
    exit 1
 fi

cd "${work_dir}"
rm -fv ./solid
cp ./build/solid .

echo "Building the tester..."
(cmake . && make) &> tester-build.log

# Specify the precice-config
ln -sf precice-config_unified.xml precice-config.xml
for i in "${tests[@]}"
    do
    test_name="${i}_serial"
    print_start "${test_name}"
    timeout 120s ./solid "$i"/"$i".prm > "$i"/"${test_name}".log 2>&1 &
    pid_solid=$!
    timeout 120s ./dummy_tester > "$i"/tester-"${test_name}".log 2>&1 &
    pid_dummy=$!
    wait $pid_solid; rc_solid=$?
    wait $pid_dummy; rc_dummy=$?
    if [ "$rc_solid" -ne 0 ] || [ "$rc_dummy" -ne 0 ]; then
        echo "Test $test_name failed (solid=$rc_solid, dummy=$rc_dummy)."
        echo "----- solid.log -----"
        cat "$i"/"${test_name}".log
        echo "----- tester.log -----"
        cat "$i"/tester-"${test_name}".log
        exit 1
    fi
    numdiff  "$i"/output "$i"/"${test_name}".output > "$i"/"${test_name}".diff
    print_result "$i"/"${test_name}".diff
    test_name="${i}_parallel"
    print_start "${test_name}"
    timeout 120s mpirun --oversubscribe -np 4 ./solid "$i"/"$i".prm > "$i"/"${test_name}".log 2>&1 &
    pid_solid=$!
    timeout 120s ./dummy_tester > "$i"/tester-"${test_name}".log 2>&1 &
    pid_dummy=$!
    wait $pid_solid; rc_solid=$?
    wait $pid_dummy; rc_dummy=$?
    if [ "$rc_solid" -ne 0 ] || [ "$rc_dummy" -ne 0 ]; then
        echo "Test $test_name failed (solid=$rc_solid, dummy=$rc_dummy)."
        echo "----- solid.log -----"
        cat "$i"/"${test_name}".log
        echo "----- tester.log -----"
        cat "$i"/tester-"${test_name}".log
        exit 1
    fi
    numdiff  "$i"/output "$i"/"${test_name}".output > "$i"/"${test_name}".diff
    print_result "$i"/"${test_name}".diff
done

# Go back to initial to command directory
cd "${start_dir}"

if [ $exit_code -eq 0 ]
then
    echo "All tests passed."
else
    echo "Errors occurred: $exit_code tests failed."
    exit 1
fi
