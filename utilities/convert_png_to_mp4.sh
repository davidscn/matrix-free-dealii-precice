#!/bin/bash
# Usage ./convert_png_to_mp4.sh -r $FRAMERATE
# This script can be used in order to convert an exported paraView png series into an mp4 video
# NOTE: The image width and height must be divisible by 2
# An optional framerate can be specified as an input argument

# Default value
FRAMERATE=10

for i in "$@"; do
    case $i in
    -r|--framerate)
	FRAMERATE="$2"
        shift # past argument
        shift # past value
        ;;
    *)
        # unknown option
        ;;
    esac
done

ffmpeg -framerate "${FRAMERATE}" -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
