#!/bin/sh

BASEDIR=$(dirname "$0")

rm -fv  "${BASEDIR}"/*.log
rm -fv  "${BASEDIR}"/*.json
rm -fvr "${BASEDIR}"/precice-run
rm -fv  "${BASEDIR}"/heat
rm -fv  "${BASEDIR}"/*/*.txt
rm -fv  "${BASEDIR}"/*/*.log
rm -fv  "${BASEDIR}"/*/*/*.vtu
rm -fv  "${BASEDIR}"/*/*.diff
rm -fv  "${BASEDIR}"/*/output
