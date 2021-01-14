#!/bin/bash

# Run from this directory
cd ${0%/*} || exit 1

rm -fv ./dummy_tester/*.log
rm -fv ./dummy_tester/*.json
rm -fvr ./dummy_tester/precice-run
rm -fv ./dummy_tester/Makefile
rm -fv ./dummy_tester/solid
rm -fv ./dummy_tester/CMakeCache.txt
rm -fv ./dummy_tester/*.cmake
rm -fvr ./dummy_tester/CMakeFiles

rm -fv ./dummy_tester/*/*.txt
rm -fv ./dummy_tester/*/*.log
rm -fv ./dummy_tester/*/*.vtu
rm -fv ./dummy_tester/*/*.diff
rm -fv ./dummy_tester/*/output
