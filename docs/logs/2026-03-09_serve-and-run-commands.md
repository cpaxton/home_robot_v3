# Commands for running and testing MuJoCo + DynaMem

Quick reference for (1) `emet serve mujoco` with Robocasa scenes, (2) **SVM (mapping)** vs **DynaMem** testing in Robocasa, and (3) `emet run dynamem` with various goals. See [simulation.md](../simulation.md), [cli.md](../cli.md), and [dynamem.md](../dynamem.md) for full docs.

---

## SVM vs DynaMem in Robocasa (overview)

| | **SVM (mapping)** | **DynaMem** |
|---|-------------------|-------------|
| **App** | `emet run mapping` | `emet run dynamem` |
| **What it does** | Explore and build the **semantic voxel map (SVM)** only. Saves a `.pkl` map you can inspect or reuse. | Explore, build semantic memory, and run **pick-and-place** from text goals. |
| **Controller** | InstanceMemoryController (RobotAgent) | DynamemController |
| **Use case** | Test mapping/exploration and semantic memory; save a map for offline tests (e.g. `read_map --show-svm`, `test_svm.py`). | End-to-end testing: “pick apple, place on plate” in Robocasa. |

Both run against the same MuJoCo server. Use **Robocasa** for richer kitchen scenes and object names (apple, bowl, can, cabinet, etc.).

---

## 1. emet serve mujoco (with Robocasa)

**Prereqs:** `emet install sim` then `emet sync -e sim` (or `emet install robocasa` then sync) so Robocasa scene generation is available.

### Basic Robocasa server

```bash
emet serve mujoco --use-robocasa
```

Default task is `PickPlaceCounterToCabinet`; default style and layout are 1.

### Robocasa with task / style / layout

Pass through to the underlying mujoco_server (use `--` before extra options if your shell needs it):

```bash
emet serve mujoco --use-robocasa --robocasa-task PickPlaceCounterToCabinet --robocasa-style 1 --robocasa-layout 1
```

### Headless (no display / EGL)

```bash
emet serve mujoco --use-robocasa --headless
```

### Ports already in use

Ports 4401–4404 are freed automatically when you start the server. If you still need different ports:

```bash
emet serve mujoco --use-robocasa --port-offset 100
# Uses 4501–4504
```

### Stop the server

```bash
emet kill-mujoco-server
# or free all default ports:
emet kill-mujoco-server --all
```

---

## 2. emet run mapping (SVM) in Robocasa

Build and save the semantic voxel map (SVM) in a Robocasa scene. Same server as DynaMem; no pick/place.

**Terminal 1 – server:**

```bash
emet serve mujoco --use-robocasa
```

**Terminal 2 – mapping (explore and save map):**

```bash
emet run mapping --robot-ip 127.0.0.1
```

Optional: set exploration steps and output filename by passing through to the app (e.g. `--explore-iter 5`, `--output-filename robocasa_map`). The mapping app saves the map to a `.pkl` file (default name from `--output-filename`, e.g. `stretch_output.pkl` when using defaults).

**Inspect the saved SVM:**

```bash
python -m emet.app.read_map -i <path/to/saved_map.pkl> --show-svm
# Or with planning tests:
python -m emet.app.read_map -i <path/to/saved_map.pkl> --show-svm --test-planning
```

**Unit tests that use SVM:** `src/test/mapping/test_svm.py` (expects pre-made `.pkl` data under `src/test/mapping/`, e.g. `hq_small.pkl` / `hq_large.pkl`). These are offline tests on saved maps, not live Robocasa.

---

## 3. emet run dynamem (with goals)

Start the MuJoCo server in one terminal (e.g. `emet serve mujoco --use-robocasa`), then in another terminal run DynaMem. For simulation, use `--robot-ip 127.0.0.1` and `--server-ip 127.0.0.1`. Use `-S` to skip confirmations for autonomous runs.

### Basic run (prompts for mode and object/receptacle)

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 --visual-servo --match-method class
```

At the mode prompt choose **E** (explore), **M** (pick and place), or **Q** (quit). For **M** you’ll be prompted for target object and receptacle.

### Run with explicit pick/place goals

Pass `--target-object` and `--target-receptacle` so you don’t have to type object/receptacle at the prompts. With `-S`, confirmations are skipped; you may still see the mode prompt (E/M/Q) once—choose **M** for pick and place.

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class \
  --target-object apple --target-receptacle plate
```

More goal examples:

```bash
# Bowl to counter
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class \
  --target-object bowl --target-receptacle counter

# Can to cabinet
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class \
  --target-object can --target-receptacle cabinet

# Cup to table
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class \
  --target-object cup --target-receptacle table
```

### Headless DynaMem (Rerun via browser)

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class --headless
```

Then open Rerun at `http://localhost:9090?url=ws://localhost:9877` (or use SSH port forwarding if remote).

### CPU-only

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class --cpu
```

### Match method for visual grounding

- `--match-method class` — class-based matching (default).
- `--match-method feature` — vision-language feature similarity.

---

## 4. Two-terminal test flow (DynaMem)

**Terminal 1 – server (Robocasa):**

```bash
emet serve mujoco --use-robocasa
```

**Terminal 2 – DynaMem with a goal:**

```bash
emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class \
  --target-object apple --target-receptacle plate
```

To exit DynaMem cleanly, use the mode prompt and choose **Q** (quit) when the app asks, or stop the process. Then stop the server with `emet kill-mujoco-server` if needed.

---

## 5. Testing checklist (SVM vs DynaMem in Robocasa)

1. **Robocasa server**  
   `emet serve mujoco --use-robocasa` (same for both flows).

2. **SVM only (mapping)**  
   - `emet run mapping --robot-ip 127.0.0.1`  
   - Optionally set `--explore-iter`, `--output-filename`.  
   - Inspect: `python -m emet.app.read_map -i <map.pkl> --show-svm`.

3. **DynaMem (pick-and-place)**  
   - `emet run dynamem --robot-ip 127.0.0.1 --server-ip 127.0.0.1 -S --visual-servo --match-method class`  
   - Add goals: `--target-object apple --target-receptacle plate` (and choose **M** at the mode prompt).

4. **Relevant tests**  
   - **DynaMem semantic memory (unit):** `pytest src/test/mapping/test_semantic_memory.py`  
   - **SVM from saved map (unit):** `pytest src/test/mapping/test_svm.py` (requires `hq_small.pkl` / `hq_large.pkl`)  
   - **Red cylinder in default sim:** `src/test/mapping/test_red_cylinder_in_sim.py` (default MuJoCo scene, not Robocasa)
