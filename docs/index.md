# Welcome to the documentation

This repository aims to implement a modular library for correcting tracking results. By tracking, we mean:

- On a sequence of images, an initial detection algorithm (e.g., yolo, fast-RCNN) is applied upstream.
- A tracking algorithm (e.g., Bytetrack, Strongsort) is then applied to the detections with the aim of assigning a unique ID to each different object and tracking these objects, i.e., maintaining the unique ID throughout the image sequence.

Overall, state-of-the-art (SOTA) tracking algorithms perform well in cases of constant speed movements, with detections not evolving (shape of bounding boxes relatively constant), which does not fit many real use cases. In practice, we end up with a lot of ID switches, and far too many unique IDs compared to the number of different objects. Therefore, we propose here a library for re-matching IDs, based on a tracking result, and allowing to reassign object IDs to ensure uniqueness.

Here is an example of the track reid library, used to correct jungling balls tracking results on a short video.

<p align="center">
  <img src="https://storage.googleapis.com/track-reid/assets/output_no_reid.gif" width="500"/><br>
  <b>Bytetrack x yolov8l, 42 tracked objects</b>
</p>
<p align="center">
  <img src="https://storage.googleapis.com/track-reid/assets/output_with_reid.gif" width="500"/><br>
  <b>Bytetrack x yolov8l + track-reid, 4 tracked objects </b>
</p>

For more insight on how to get started, please refer to [this guide for users](quickstart_user.md), or [this guide for developers](quickstart_dev.md).
