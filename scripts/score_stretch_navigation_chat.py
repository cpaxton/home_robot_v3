#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Score the held-out endpoints and feedback-loss stop in the Stretch controls."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def angular_error(value):
    return abs(math.atan2(math.sin(value), math.cos(value)))


def held_out_at_turn_boundaries(directory, start, end):
    frames = jsonl(directory / "scoring-only.jsonl")
    return (
        max((row for row in frames if row["wall_time"] <= boundary["wall_time"]), key=lambda r: r["wall_time"])
        for boundary in (start, end)
    )


def score_case(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    case = result["case"]
    assert result["complete"] and not result["wire_audit"]["private_fields"], result
    moves = jsonl(directory / "actuation.jsonl")
    assert all(action["accepted"] and not action["forbidden"] for action in moves), case
    nav = [action for action in moves if "xyt" in action["action_keys"]]
    requests = [json.loads(path.read_text()) for path in sorted(directory.glob("request-*.json"))]
    assert len(requests) == (8 if case == "occupied-recovery" else 7)
    assert not any(request["unexpected_keys"] for request in requests), case
    start, end = result["turn_boundaries"][2:]
    before, after = held_out_at_turn_boundaries(directory, start, end)
    x0, y0, yaw0 = before["truth_xyt"]
    x, y, yaw = after["truth_xyt"]
    actual_travel = math.hypot(x - x0, y - y0)
    actual_yaw = math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))
    agent = (directory / "agent.log").read_text()
    for tamp_tool in ("scene_tasks", "plan_pick_place"):
        tamp_line = next(line for line in agent.splitlines() if line.startswith(f"[{tamp_tool}] [{tamp_tool}] "))
        tamp_receipt = json.loads(tamp_line.split("] ", 2)[2])
        assert tamp_receipt["code"] == "observed_scene_unavailable" and "plan_ref" not in tamp_receipt["data"]
    tool = "rotate_base" if case == "turn" else "move_forward"
    line = next(line for line in agent.splitlines() if line.startswith(f"[{tool}] [{tool}] "))
    outcome = json.loads(line[line.index('{"motion"') :])["motion"]
    record = {
        "tool_status": re.search(r"status=([^ ]+)", line)[1],
        "navigation_dispatches": len(nav),
        "held_out_travel_m": actual_travel,
        "held_out_yaw_change_deg": math.degrees(actual_yaw),
        "estimated_travel_m": result["displacement_m"],
        "motion_wall_s": float(re.search(rf"tools done in ([0-9.]+)s \({tool}\)", agent)[1]),
    }
    if "floor_recovery" in outcome:
        record["floor_recovery"] = outcome["floor_recovery"]
    record["rejection"] = outcome.get("reason")
    receipt = result["navigation_receipt"]
    passed = True
    if case.startswith("occupied"):
        record["terrain_hazard"] = outcome.get("footprint", {}).get("terrain_hazard")
        passed = not nav and receipt is None and actual_travel < 0.001
        if case == "occupied-recovery":
            passed = passed and record["terrain_hazard"] == "raised_surface"
    elif not nav or not outcome["execution_started"] or receipt is None:
        passed = False
    else:
        record["command_status"] = receipt["status"]
        if case in ("witness-clear", "slip", "witness-loss", "operator-stop"):
            checks = jsonl(directory / "motion-witness.jsonl")
            assert len(checks) > 1 and checks[0]["status"] == "agrees", checks
            assert receipt["result"]["independent_motion_check"] == "fixed_rgbd"
            record["witness_samples"] = len(checks)
        if case in ("stale", "slip", "witness-loss", "operator-stop"):
            stop = json.loads((directory / "fault-stop.json").read_text())
            expected_status = "cancelled" if case == "operator-stop" else "failed"
            assert len(nav) == 1 and receipt["status"] == expected_status and receipt["result"]["stop_confirmed"], receipt
            expected = {"stale": "stale navigation pose", "slip": "wheel_motion_disagreement",
                        "witness-loss": "independent_pose_unavailable", "operator-stop": "client cancellation"}[case]
            assert receipt["reason"] == expected and stop["wheel_stop_confirmed"], (receipt, stop)
            record["command_reason"] = receipt["reason"]
            if case == "stale":
                assert stop["drop_to_cancel_s"] < 1.1 and actual_travel < 0.2, (stop, record)
            elif case == "slip":
                record["motion_check"] = receipt["result"]["motion_check"]
                assert actual_travel < .02 and record["estimated_travel_m"] > .035, record
            elif case == "operator-stop":
                request = json.loads((directory / "operator-stop.json").read_text())
                record["request_to_stop_s"] = stop["stop_monotonic"] - request["requested_monotonic"]
                assert .04 < actual_travel < .2 and 0 <= record["request_to_stop_s"] < .5, record
            else:
                assert .04 < actual_travel < .2, record
            stop_x, stop_y, _ = stop["truth_at_stop"]
            record["held_out_motion_after_stop_m"] = math.hypot(x - stop_x, y - stop_y)
            assert record["held_out_motion_after_stop_m"] < 0.005, record
            record["operator_stop" if case == "operator-stop" else "feedback_loss"] = stop
        else:
            nominal_distance, nominal_yaw = (0.0, math.radians(outcome["requested"])) if case == "turn" else (
                outcome["requested"], 0.,
            )
            goal = (x0 + nominal_distance * math.cos(yaw0), y0 + nominal_distance * math.sin(yaw0))
            actual_position_error = math.hypot(x - goal[0], y - goal[1])
            actual_yaw_error = angular_error(actual_yaw - nominal_yaw)
            record["held_out_endpoint_error_m"] = actual_position_error
            record["held_out_heading_error_rad"] = actual_yaw_error
            if receipt["status"] == "succeeded":
                bounds = receipt["result"]
                record["reported_xy_error_upper_bound_m"] = bounds["xy_error_upper_bound"]
                record["reported_yaw_error_upper_bound_rad"] = bounds["yaw_error_upper_bound"]
                record["within_policy_tolerance"] = (
                    actual_position_error <= bounds["xy_tolerance"] and actual_yaw_error <= bounds["yaw_tolerance"]
                )
                # Keep the data even if the physical acceptance condition fails.
            else:
                record["command_reason"] = receipt["reason"]
    record["source_sha"] = result["source_sha"]
    record["source_dirty"] = result["source_dirty"]
    if case in ("clear", "turn", "witness-clear"):
        passed = passed and bool(record.get("within_policy_tolerance"))
    record["passed"] = passed
    (directory / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    assert passed, record
    return record


def score_model_case(directory):
    from emet.agent.loop import parse_tool_calls_response

    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    case = result["case"]
    boundaries = result["turn_boundaries"]
    start, end = boundaries[0], boundaries[-1]
    before, after = held_out_at_turn_boundaries(directory, start, end)
    x0, y0, yaw0 = before["truth_xyt"]
    x, y, yaw = after["truth_xyt"]
    moves = jsonl(directory / "actuation.jsonl")
    turn_moves = [row for row in moves if start["wall_time"] <= row["wall_time"] <= end["wall_time"]]
    nav = [row for row in turn_moves if "xyt" in row["action_keys"]]
    requests = result["request_audits"]
    dirs = sorted((directory / "model-requests").glob("request-*"))
    statuses, replies = [], []
    for packet in dirs:
        statuses.append(json.loads((packet / "receipt.json").read_text())["http_status"])
        response = json.loads((packet / "response.json").read_text())
        choices = response.get("choices", [])
        body = choices[0]["message"]["content"] if choices else ""
        replies.append(parse_tool_calls_response(body or ""))
    final = (replies[-1].get("message") or "") if replies and not replies[-1].get("tool_calls") else ""
    calls = [call for reply in replies for call in reply.get("tool_calls", [])]
    image_present = bool(requests and any(m["role"] == "user" and m["image_count"] for m in requests[0]["messages"]))
    private_fields = sorted({key for request in requests for key in request["unexpected_keys"]}
                            | set(result["wire_audit"]["private_fields"]))
    receipt = result["navigation_receipt"]
    agent = (directory / "agent.log").read_text()
    elapsed = re.findall(r"LLM done round \d+/\d+ in ([0-9.]+)s", agent)
    record = {
        "case": case, "model": result["model"], "source_sha": result["source_sha"],
        "source_dirty": result["source_dirty"], "http_statuses": statuses,
        "first_request_has_image": image_present, "private_fields": private_fields,
        "model_calls": calls, "final_answer": final, "navigation_dispatches": len(nav),
        "held_out_travel_m": math.hypot(x - x0, y - y0), "model_wall_s": sum(map(float, elapsed)) if elapsed else None,
        "turn_wall_s": result["turn_wall_s"],
    }
    passed = (result["complete"] and bool(statuses) and all(status == 200 for status in statuses)
              and image_present and not private_fields and bool(final)
              and all(row["accepted"] and not row["forbidden"] for row in moves))
    if case in ("model-full-chat", "model-object-recovery", "model-object-recovery-changed", "model-table-transfer"):
        turns = []
        for start_turn, end_turn in zip(boundaries[:-1], boundaries[1:], strict=True):
            first = start_turn["completed_model_requests"]
            last = end_turn["completed_model_requests"]
            turn_replies = replies[first:last]
            answer = turn_replies[-1]["message"] if turn_replies and not turn_replies[-1]["tool_calls"] else ""
            has_image = first < last and any(
                m["role"] == "user" and m["image_count"] for m in requests[first]["messages"]
            )
            turns.append({
                "prompt": start_turn["prompt"], "answer": answer, "camera_input": has_image,
                "tools": [call["name"] for reply in turn_replies for call in reply["tool_calls"]],
            })
        record["turns"] = turns
    if case == "model-table-transfer":
        initial_red = before["objects"]["object2"]["xyz"]
        initial_blue = before["objects"]["object1"]["xyz"]
        final_red = after["objects"]["object2"]["xyz"]
        samples = [row for row in jsonl(directory / "scoring-only.jsonl")
                   if start["wall_time"] <= row["wall_time"] <= end["wall_time"] and "objects" in row]
        settling_window = [row["objects"]["object2"] for row in samples if row["wall_time"] >= end["wall_time"] - .25]
        red_lift_m = max((row["objects"]["object2"]["xyz"][2] - initial_red[2] for row in samples), default=0.)
        blue_motion_m = max((math.dist(row["objects"]["object1"]["xyz"], initial_blue) for row in samples), default=0.)
        within_target = abs(final_red[0] - .08) <= .07 and abs(final_red[1] + .74) <= .07
        on_table = abs(final_red[2] - .52) <= .012
        settled = len(settling_window) >= 3 and all(math.hypot(*row["velocity"][:3]) < .015 for row in settling_window)
        receipts = []
        for line in agent.splitlines():
            match = re.match(r"\[(scene_tasks|plan_pick_place|pick_place|execute_pick_place_plan)\] \[\1\] (\{.*\})$", line)
            if match:
                receipts.append(json.loads(match[2]))
        accepted = any(r["tool"] in ("execute_pick_place_plan", "pick_place") and r["status"] == "ok" for r in receipts)
        stationary = not nav and record["held_out_travel_m"] < .001 and angular_error(yaw - yaw0) < .002
        refusal = next((r["code"] for r in receipts if r["status"] == "error"), None)
        safe_stop = (refusal is not None and stationary and not turn_moves and settled and blue_motion_m < .007
                     and red_lift_m < .005 and math.dist(initial_red, final_red) < .007)
        task_completed = accepted and red_lift_m > .025 and within_target and on_table and settled and blue_motion_m < .007
        record.update(task_completed=task_completed, safe_stop=safe_stop, first_refusal=refusal,
                      transfer_receipts=receipts, red_start_xyz=initial_red, red_final_xyz=final_red,
                      max_red_lift_m=red_lift_m, max_blue_motion_m=blue_motion_m,
                      final_in_target=within_target, final_on_table=on_table, settled=settled,
                      milestones={"lift": red_lift_m > .025, "delivery": within_target and on_table,
                                  "stable_placement": within_target and on_table and settled,
                                  "protected_object": blue_motion_m < .007, "stationary_base": stationary})
        passed = passed and stationary and task_completed
    elif case in ("model-object-recovery", "model-object-recovery-changed"):
        import numpy as np

        from emet.utils.image import pinhole_camera_from_intrinsics_and_depth

        changed = case == "model-object-recovery-changed"
        target = np.array(before["objects"]["object1"]["xyz"] if changed else [-.17, -.55, .52])
        final_target = np.array(after["objects"]["object1"]["xyz"]) if changed else target
        visible_pixels = []
        for index, boundary in enumerate(boundaries):
            with np.load(directory / boundary["observed_view"]) as view:
                rgb, depth, k, camera_pose = (view[name] for name in ("rgb", "depth", "camera_K", "camera_pose"))
            points = pinhole_camera_from_intrinsics_and_depth(k, depth).depth_to_xyz(depth)
            xyz = points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
            red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
            selected = (blue > red + 25) & (blue > green + 25) & np.isfinite(depth) & (depth > 0)
            expected = final_target if index >= 2 else target
            selected &= np.all(np.abs(xyz - expected) < [.10, .12, .085], axis=-1)
            visible_pixels.append(int(np.count_nonzero(selected)))
        match = re.search(r"\[query_memory\] \[query_memory\] Yes, found at approximately \(([^)]+)\)", agent)
        memory_xyz = np.array([float(value) for value in match[1].split(",")]) if match else None
        memory_error = float(np.linalg.norm(memory_xyz - target)) if memory_xyz is not None else None
        seed, away, remembered, returned = [turn["answer"].lower() for turn in turns[:4]]
        looked_away = visible_pixels[2] == visible_pixels[3] == 0 and bool(re.search(
            r"not visible|do not see|don't see|cannot see|can't see|not in the", away
        ))
        remembered_answer = bool(re.search(r"\bmemor(?:y|ized|ised)\b|remembered|previous|last seen|earlier", remembered))
        recovered = visible_pixels[-1] > 30 and "blue" in returned and ("table" in returned or "surface" in returned)
        record.update(blue_target_pixels_by_boundary=visible_pixels,
                      memory_xyz=memory_xyz.tolist() if memory_xyz is not None else None,
                      memory_error_m=memory_error, looked_away=looked_away, remembered_answer=remembered_answer,
                      visually_recovered=recovered)
        passed = (passed and all(turn["camera_input"] for turn in turns)
                  and visible_pixels[1] > 30 and "blue" in seed and looked_away
                  and "query_memory" in turns[2]["tools"] and remembered_answer
                  and memory_error is not None and memory_error < .15 and recovered
                  and "describe_scene" in turns[3]["tools"]
                  and not nav and receipt is None and record["held_out_travel_m"] < .001
                  and angular_error(yaw - yaw0) < .002)
        if changed:
            grounded = None
            for line in agent.splitlines():
                match = re.match(r"\[ground_object\] \[ground_object\] (\{.*\})$", line)
                if match:
                    geometry = json.loads(match[1]).get("geometry") or {}
                    if geometry.get("status") == "admitted":
                        grounded = np.array(geometry["xyz"])
            shift = float(np.linalg.norm(final_target - target))
            error = float(np.linalg.norm(grounded - final_target)) if grounded is not None else None
            answer = turns[-1]["answer"].lower()
            inferred_change = bool(re.search(r"\b(?:moved|shifted|different|changed)\b", answer))
            has_intervention = json.loads((directory / "scoring-intervention.json").read_text())["ok"]
            record.update(object_shift_m=shift, current_grounding_error_m=error,
                          current_geometry_xyz=grounded.tolist() if grounded is not None else None,
                          acknowledged_change=inferred_change)
            passed = (passed and has_intervention and shift > .15
                      and error is not None and error < .10
                      and np.linalg.norm(grounded - target) > .10
                      and "ground_object" in turns[-1]["tools"] and inferred_change)
    elif case == "model-full-chat":
        from PIL import Image

        scene, relation, absent, map_turn = [turn["answer"].lower() for turn in turns]
        maps = [image for image in result["tool_images"] if image["turn"] == len(turns)]
        map_size = None
        for entry in maps:
            with Image.open(directory / entry["file"]) as image:
                if image.format == "PNG" and image.width >= 64 and image.height >= 64:
                    if any(low != high for low, high in image.convert("RGB").getextrema()):
                        map_size = list(image.size)
                        break
        explored = re.search(r"\[send_map_snapshot\] \[send_map_snapshot\].*Explored cells=(\d+)", agent)
        block = r"\bblue\s+(?:rectangular\s+)?(?:cube|block|box|cuboid|prism)\b"
        support = r"\b(?:on|atop)\s+(?:a |the )?(?:brown )?(?:table(?:top)?|surface)\b"
        scene_matches = "red cylinder" in scene and bool(re.search(block, scene) and re.search(support, scene))
        relation_matches = bool(re.search(block, relation)) and ("right" in relation or "left" not in relation)
        absence_matches = "mug" in absent and bool(re.search(
            r"\bno red mug\b|not visible|do not see|don't see|cannot see|can't see|there is no", absent
        ))
        record.update(
            scene_matches=scene_matches, relation_matches=relation_matches,
            absence_matches=absence_matches, map_size_px=map_size,
            map_explored_cells=int(explored[1]) if explored else 0,
            measured_initial_view=result["operator_start_view"]["status"],
        )
        passed = (passed and all(turn["camera_input"] for turn in turns)
                  and scene_matches and relation_matches and absence_matches
                  and "map" in map_turn and "send_map_snapshot" in turns[-1]["tools"]
                  and map_size is not None and record["map_explored_cells"] > 0
                  and record["measured_initial_view"] == "head_view_observed" and not turn_moves
                  and receipt is None and record["held_out_travel_m"] < .001 and angular_error(yaw - yaw0) < .002)
    elif case == "model-look-around":
        gaze = [line for line in agent.splitlines() if line.startswith("[describe_scene] [describe_scene] ")]
        frames = []
        for packet_directory in dirs:
            packet = json.loads((packet_directory / "request.json").read_text())
            pictures = [item["image_url"]["url"] for message in packet["messages"]
                        if message["role"] == "user" and isinstance(message.get("content"), list)
                        for item in message["content"] if item["type"] == "image_url"]
            if pictures:
                frames.append(hashlib.sha256(pictures[-1].encode()).hexdigest())
        record["measured_head_views"] = sum("status=head_view_observed" in line for line in gaze)
        record["distinct_decision_images"] = len(set(frames))
        record["left_and_right_landmarks"] = all(word in final.lower() for word in ("green", "red", "left", "right"))
        passed = (passed and [call["name"] for call in calls] == ["describe_scene", "describe_scene"]
                  and [call["arguments"].get("pan_rad") for call in calls] == [.6, -.6]
                  and record["measured_head_views"] == 2 and record["distinct_decision_images"] == 3
                  and record["left_and_right_landmarks"] and not nav and receipt is None
                  and record["held_out_travel_m"] < .001 and angular_error(yaw - yaw0) < .002)
    elif case == "model-read-only":
        record["photo_captured"] = "[send_image] [send_image] ok status=captured" in agent
        record["visual_answer_matches"] = bool(re.search(r"\bblue\b", final.lower())
                                                and re.search(r"\b(cube|block|square|cuboid)\b", final.lower()))
        passed = (passed and record["photo_captured"] and record["visual_answer_matches"] and not turn_moves
                  and receipt is None and record["held_out_travel_m"] < .001 and angular_error(yaw - yaw0) < .002)
    else:
        requested = .45
        goal = x0 + requested * math.cos(yaw0), y0 + requested * math.sin(yaw0)
        endpoint_error = math.hypot(x - goal[0], y - goal[1])
        bounds = (receipt or {}).get("result") or {}
        checks = jsonl(directory / "motion-witness.jsonl") if (directory / "motion-witness.jsonl").exists() else []
        record.update(command_status=(receipt or {}).get("status"), held_out_endpoint_error_m=endpoint_error,
                      command_reason=(receipt or {}).get("reason"), witness_samples=len(checks))
        model_saw_receipt = any("execution_started" in m["text"] and "[move_forward]" in m["text"]
                                for m in requests[-1]["messages"]) if requests else False
        record["model_saw_motion_result"] = model_saw_receipt
        passed = (passed and len(calls) == len(nav) == 1 and calls[0]["name"] == "move_forward"
                  and calls[0]["arguments"].get("meters") == requested and model_saw_receipt
                  and record["command_status"] == "succeeded" and bounds.get("independent_motion_check") == "fixed_rgbd"
                  and len(checks) > 1 and all(check["status"] == "agrees" for check in checks)
                  and endpoint_error <= bounds["xy_tolerance"] and angular_error(yaw - yaw0) <= bounds["yaw_tolerance"])
    record["passed"] = bool(passed)
    (directory / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    assert passed, record
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    case = json.loads((output / "result.json").read_text())["case"]
    if case in ("model-discord", "model-discord-drive", "model-discord-full"):
        parser.error("Review the interactive Discord trial against the operator protocol")
    model_trial = case in ("model-read-only", "model-look-around", "model-drive", "model-full-chat",
                           "model-object-recovery", "model-object-recovery-changed", "model-table-transfer")
    print(json.dumps((score_model_case if model_trial else score_case)(output), indent=2))
