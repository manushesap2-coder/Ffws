#!/bin/bash
# usage: mk_inter.sh src out start dur [nice]
L=/tmp/jobs/lut/mini4pro_dlogm.cube
src=$1; out=$2; ss=$3; dur=$4
VF="scale=1296:2304:flags=bicubic:in_color_matrix=bt709:in_range=tv,format=rgb48le,lut3d=file=$L:interp=tetrahedral,scale=out_color_matrix=bt709:out_range=tv,format=yuv422p10le"
ffmpeg -v error -y -ss $ss -t $dur -i "$src" -an -vf "$VF" -c:v libx264 -preset ultrafast -crf 7 -g 30 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv "$out" && echo "DONE $out"
