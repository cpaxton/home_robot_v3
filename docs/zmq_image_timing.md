# Image timing and review acceptance

Mars publishes `head_cam_left/image_timing`, `head_cam_right/image_timing`,
and `ee_cam/image_timing` alongside their pixels on full and servo streams.
The optional H.264 stream carries the corresponding left-camera timing.
Each record contains `timestamp_ns`, `clock_domain: ros`,
`source: image_header`, and `available` (whether a real frame was received).
The timestamp and copied pixels are read under one camera lock. Republished
frames retain their original stamp. Zero or absent stamps are `null`, not
publication time. Missing-camera black placeholders have `available: false`.

Generic clients retain these records in `Observations.image_timing`, keyed
by camera name. Legacy peers have no timing records. Servo-to-full merging
copies timing only when it copies the corresponding pixels, clearing any
unrelated existing timing when the source has none.

## What this does not establish

A ROS image header stamp is the driver's timestamp, not proof of hardware
exposure time. ROS time may be simulated, reset, or unsynchronized with the
receiver. Do not subtract it from workstation wall time. Repeated stamps
can detect repeated samples; backward jumps require resetting freshness
history. A local monotonic receive timer can detect stalled delivery but
does not establish acquisition age. Pose, calibration, depth and embeddings
are not yet certified as synchronized with these images.

## PR #135 review checklist

- [x] Atomic Mars image/header timestamp snapshots; no synthetic capture times.
- [x] Full, servo and H.264 publication; generic-client preservation.
- [x] Metadata-only merging preserves image/timestamp pairing.
- [x] Tests for repeated, unknown and missing frames and legacy peers.
- [ ] Stationary hardware test: inspect actual header progression, duplicates
  and clock resets; verify driver timestamp source before claiming capture age.
- [ ] Test left/right stamp skew before treating stereo frames as synchronized.
- [ ] Bind derived depth/embedding outputs to their source frames, and use
  timestamped TF lookup before claiming geometry is acquisition-aligned.
- [ ] Extend the same documented contract to Stretch and simulator publishers,
  explicitly identifying simulation clock domains.
- [ ] Add bounded stale-frame rejection at consumers, with explicit unknown
  handling rather than treating legacy streams as fresh.

The first four items are this patch's scope. Remaining items are follow-ups,
not advertised capabilities. Review the broader PR's deployment, RTSP,
compression and stream-load tests separately before merging. Do not overwrite
a newer installed robot bridge to test this patch; compare versions first.
