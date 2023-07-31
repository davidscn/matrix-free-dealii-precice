#!/bin/sh

BASEDIR=$(dirname "$0")
"${BASEDIR}"/solid/run_tests.sh
"${BASEDIR}"/heat/run_tests.sh
