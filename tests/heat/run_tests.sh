#!/bin/bash
set -e
echo "Testing the heat transfer"
RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

start_dir=$(pwd)

# Change to the working directory
cd "$(dirname "$0")"
work_dir=$(pwd)

# Declare test names
declare -a tests=("partitioned-heat")

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
	fi
}

test_name="building"
print_start ${test_name}
mkdir -p "${work_dir}"/build && cd "${work_dir}"/build
(cmake ../../../ && make debug && make heat) >${test_name}.log
 if [ $? -eq 0 ]
    then
    echo -ne "${GREEN} passed ${NOCOLOR}\n"
    else
    echo -ne "${RED} failed ${NOCOLOR}\n"
    exit 1
 fi

cd "${work_dir}"
rm -fv ./heat
cp ./build/heat .

# Specify the precice-config
for i in "${tests[@]}"
    do
    test_name="${i}_serial"
    print_start "${test_name}"
    ./heat "$i"/dirichlet.prm > "$i"/dirichlet.log & ./heat "$i"/neumann.prm > "$i"/neumann.log
    grep 'Error' "$i"/dirichlet.log > "$i"/output
    grep 'Error' "$i"/neumann.log >> "$i"/output
    numdiff  "$i"/output "$i"/"${test_name}".output > "$i"/"${test_name}".diff
    print_result "$i"/"${test_name}".diff
    test_name="${i}_parallel"
    print_start "${test_name}"
    mpirun --oversubscribe -np 4 ./heat "$i"/dirichlet.prm > "$i"/dirichlet.log & mpirun --oversubscribe -np 4 ./heat "$i"/neumann.prm > "$i"/neumann.log
    grep 'Error' "$i"/dirichlet.log > "$i"/output
    grep 'Error' "$i"/neumann.log >> "$i"/output
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
