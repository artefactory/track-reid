#!/bin/bash

N_SEQUENCES=5

sequences_detections=$(gsutil ls gs://data-track-reid/detections | head -$N_SEQUENCES)
sequences_frames=$(gsutil ls gs://data-track-reid/frames| head -$N_SEQUENCES)

# remove first sequence which is the bucket name
sequences_detections=$(echo "$sequences_detections" | tail -n +2)
sequences_frames=$(echo "$sequences_frames" | tail -n +2)

# download the sequences to data/detections and data/frames
for sequence in $sequences_detections; do
    # Extract the sequence number from the sequence string
    sequence_number=$(echo $sequence | awk -F'/' '{print $5}')
    mkdir -p data/detections/$sequence_number
    # Get only the .txt files from the sequence and download them
    gsutil -m cp -r $(gsutil ls $sequence | grep '.txt$') data/detections/$sequence_number || echo "Failed to download $sequence_number, skipping..."
done

for sequence in $sequences_frames; do
    # Extract the sequence number from the sequence string
    sequence_number=$(echo $sequence | awk -F'/' '{print $5}')
     mkdir -p data/frames/$sequence_number
    # Get only the .jpg files from the sequence and download them
    gsutil -m cp -r $(gsutil ls $sequence | grep '.jpg$') data/frames/$sequence_number || echo "Failed to download $sequence_number, skipping..."
done
