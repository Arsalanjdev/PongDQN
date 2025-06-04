#!/bin/bash

for f in *.mp4; do
    filename="${f%.*}"
    ffmpeg -i "$f" -vf "fps=10,scale=320:-1" -loop 0 "${filename}.webp"
done
