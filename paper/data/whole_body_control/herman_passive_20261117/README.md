# Herman passive audit and offline reader patch validation

Actual robot source revision 0d4c350d; arm ~200 Hz, odometry ~30 Hz over a
six-second passive subscription. Correct middleware is rmw_zenoh_cpp/domain 0.
The active source retains the known velocity-unit and failed-read defects.
The current-source patch passes local temporary-tree C++ SDK-double checks;
no robot files, services, motor modes or commands were changed. Raw copied
source stays ignored. See the [audit](../../../../docs/experiments/herman_feedback_audit_20261117.md).
