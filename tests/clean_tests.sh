#!/bin/sh

BASEDIR=$(dirname "$0")
"${BASEDIR}"/solid/clean_tests.sh
"${BASEDIR}"/heat/clean_tests.sh
