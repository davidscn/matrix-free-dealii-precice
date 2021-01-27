# Usage:
#  utilities/run_clang_tidy.sh SRC_DIR OPTIONAL_CMAKE_ARGS
#   with:
#     SRC_DIR points to the main directory
#     OPTIONAL_CMAKE_ARGS are optional arguments to pass to CMake
#   make sure to run this script in an empty build directory
#
# Requirements:
# Clang 5.0.1+ and have clang, clang++, and run-clang-tidy.py in
# your path.

# grab first argument and make relative path an absolute one:
SRC=$1
SRC=$(cd "$SRC";pwd)
shift

echo "SRC-DIR=$SRC"

# export compile commands (so that run-clang-tidy.py works)
ARGS=("-D" "CMAKE_EXPORT_COMPILE_COMMANDS=ON" "-D" "CMAKE_BUILD_TYPE=Debug" "-D" "CMAKE_CXX_COMPILER=clang++" "$@")

# for a list of checks, see /.clang-tidy
cat "$SRC/.clang-tidy"

if ! [ -x "$(command -v run-clang-tidy)" ] || ! [ -x "$(command -v clang++)" ]; then
    echo "make sure clang, clang++, and run-clang-tidy (part of clang) are in the path"
    exit 1
fi

mkdir -p ${SRC}/clang-tidy
cd ${SRC}/clang-tidy
echo `pwd`
cmake "${ARGS[@]}" "$SRC" || (echo "cmake failed!"; false) || exit 2

# pipe away stderr (just contains nonsensical "x warnings generated")
# pipe output to output.txt
run-clang-tidy -p . -quiet -header-filter "$SRC/include/*" -extra-arg='-DCLANG_TIDY' 2>error.txt >output.txt

# grep interesting errors and make sure we remove duplicates:
grep -E '(warning|error): ' output.txt | sort | uniq >clang-tidy.log

# if we have errors, report them and set exit status to failure
if [ -s clang-tidy.log ]; then
    cat clang-tidy.log
    exit 3
fi

echo "All passed"
exit 0

