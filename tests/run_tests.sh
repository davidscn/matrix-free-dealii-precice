#!/bin/bash
echo "Running tests for matrix-free-dealii-precice"
RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

# Run from this directory
cd ${0%/*} || exit 1

# Declare test names
declare -a tests=("turek_gmg_scalar_referential" "turek_gmg_tensor2" "turek_gmg_tensor4" "turek_gmg_tensor4_ns" "turek_gmg_scalar" "turek_gmg_scalar_form0" "turek_jacobi_tensor2")

exit_code=0

test_dir=$(pwd)

print_start(){
	echo -e " ***Test $@:"
	echo -ne "\t Starting ..."
}

print_result() {
	if [ $? -eq 0 ]
	then
            echo -ne "${GREEN} passed ${NOCOLOR}\n"
	else
            echo -ne "${RED} failed ${NOCOLOR}\n"
            exit_code=$[$exit_code +1]
            cat $@
	fi
}

test_name="building"
print_start ${test_name}
mkdir -p build && cd build
(cmake ../../ && make debug) &>${test_name}.log
print_result ${test_name}.log
cd ..
cp ./build/solid ./dummy_tester

echo "Building the tester..."
cd ./dummy_tester
(cmake . && make) &> tester-build.log

# Specify the precice-config
ln -sf precice-config_unified.xml precice-config.xml
for i in "${tests[@]}"
    do
    test_name="${i}_serial"
    print_start ${test_name}
    ./solid $i/$i.prm&> $i/${test_name}.log & ./dummy_tester &>$i/tester-${test_name}.log
    numdiff  $i/output $i/${test_name}.output &>$i/${test_name}.diff
    print_result ${test_name}
    test_name="${i}_parallel"
    print_start ${test_name}
    mpirun -np 4 ./solid $i/$i.prm&> $i/${test_name}.log & ./dummy_tester &>$i/tester-${test_name}.log
    numdiff  $i/output $i/${test_name}.output &>$i/${test_name}.diff
    print_result $i/${test_name}.diff
done

if [ $exit_code -eq 0 ]
then
    echo "All tests passed."
else
    echo "Errors occurred: $exit_code tests failed."
    exit 1
fi
