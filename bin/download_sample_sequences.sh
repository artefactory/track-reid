#!/bin/bash

N_SEQUENCES=5

sequences_detections=$(gsutil ls gs://data-track-reid/detections | head -$N_SEQUENCES)
sequences_frames=$(gsutil ls gs://data-track-reid/frames | head -$N_SEQUENCES)

# remove first sequence which is the bucket name
sequences_detections=$(echo "$sequences_detections" | tail -n +2)
sequences_frames=$(echo "$sequences_frames" | tail -n +2)

mkdir -p data/detections
mkdir -p data/frames

# download the sequences to data/detections and data/frames
for sequence in $sequences_detections; do
    gsutil -m cp -r $sequence data/detections
done

for sequence in $sequences_frames; do
    gsutil -m cp -r $sequence data/frames
done
