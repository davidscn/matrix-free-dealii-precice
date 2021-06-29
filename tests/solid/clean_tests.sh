#!/bin/sh

BASEDIR=$(dirname "$0")

rm -fv  "${BASEDIR}"/*.log
rm -fv  "${BASEDIR}"/*.json
rm -fvr "${BASEDIR}"/precice-run
rm -fv  "${BASEDIR}"/Makefile
rm -fv  "${BASEDIR}"/solid
rm -fv  "${BASEDIR}"/CMakeCache.txt
rm -fv  "${BASEDIR}"/*.cmake
rm -fvr "${BASEDIR}"/CMakeFiles
rm -fv  "${BASEDIR}"/*/*.txt
rm -fv  "${BASEDIR}"/*/*.log
rm -fv  "${BASEDIR}"/*/*.vtu
rm -fv  "${BASEDIR}"/*/*.diff
rm -fv  "${BASEDIR}"/*/output
