"""Offline visual evidence: synchronized HTML, Rerun, MP4 and paper figures."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from pathlib import Path

from .recording import load_run, write_json


def _map_svg(scene: dict, event: dict, trajectory: list) -> str:
    from .fixture import obstacles

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 480" role="img" aria-label="Evaluator floor plan">'
    ]

    xmin = min(r["bounds"][0] for r in scene["rooms"])
    xmax = max(r["bounds"][2] for r in scene["rooms"])
    xscale = 1104 / (xmax - xmin)

    def xy(x, y):
        return 45 + (x - xmin) * xscale, 400 - (y + 2) * 78

    for room in scene["rooms"]:
        x0, y0, x1, y1 = room["bounds"]
        x, y = xy(x0, y1)
        out.append(
            f'<rect x="{x}" y="{y}" width="{(x1 - x0) * xscale}" height="{(y1 - y0) * 78}" fill="#eef2f5" stroke="#bbc5cf"/>'
        )
        out.append(f'<text x="{x + 15}" y="{y + 28}" font-size="19" fill="#334155">{html.escape(room["id"])}</text>')
    for x0, y0, x1, y1 in obstacles(scene):
        x, y = xy(x0, y1)
        out.append(f'<rect x="{x}" y="{y}" width="{(x1 - x0) * xscale}" height="{(y1 - y0) * 78}" fill="#667788"/>')
    if trajectory:
        points = " ".join(f"{x},{y}" for x, y in (xy(*p[:2]) for p in trajectory))
        out.append(f'<polyline points="{points}" fill="none" stroke="#a16207" stroke-width="3"/>')
    positions = event.get("evaluator", {}).get("positions", {})
    for name, pos in positions.items():
        obj = scene["objects"][name]
        x, y = xy(*pos[:2])
        color = "#" + "".join(f"{round(c * 255):02x}" for c in obj["color"][:3])
        if obj["movable"]:
            out.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}" stroke="white" stroke-width="2"/>')
        else:
            out.append(f'<rect x="{x - 14}" y="{y - 12}" width="28" height="24" fill="{color}" stroke="white"/>')
        # Label placements above objects and receptacles below to avoid overlap.
        lane = sum(
            other != name and scene["objects"][other]["movable"] and p[0] <= pos[0] and abs(p[0] - pos[0]) < 0.9
            for other, p in positions.items()
        )
        offset = -30 - lane * 28 if obj["movable"] else 30
        label = obj["label"].split(" ", 1)
        out.append(
            f'<text x="{x}" y="{y + offset}" text-anchor="middle" font-size="12" fill="#172033">'
            + "".join(
                f'<tspan x="{x}" dy="{0 if i == 0 else 13}">{html.escape(part)}</tspan>' for i, part in enumerate(label)
            )
            + "</text>"
        )
    pose = event.get("evaluator", {}).get("robot_xyt")
    if pose:
        x, y = xy(*pose[:2])
        radius = scene["footprint_radius_m"]
        out.append(
            f'<ellipse cx="{x}" cy="{y}" rx="{radius * xscale}" ry="{radius * 78}" fill="#0891b233" stroke="#087f9c" stroke-width="2"/>'
        )
        out.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 25 * math.cos(pose[2])}" y2="{y - 25 * math.sin(pose[2])}" stroke="#087f9c" stroke-width="4"/>'
        )
    out.append(
        f'<path d="M55 445h{xscale}m-{xscale} -5v10m{xscale} -10v10" stroke="#334155"/><text x="85" y="470" font-size="14">1 m</text>'
    )
    out.append(
        '<text x="260" y="455" font-size="16" fill="#334155">Evaluator truth · amber: executed route · cyan: robot footprint</text></svg>'
    )
    return "".join(out)


def export_html(root: Path, manifest: dict, events: list, metrics: dict):
    scene = manifest["suite"]["scene"]
    trajectory = []
    rows = []
    for event in events:
        pose = event.get("evaluator", {}).get("robot_xyt")
        if pose:
            trajectory.append(pose)
        row = dict(event)
        row["map_svg"] = _map_svg(scene, event, trajectory)
        memory = event.get("policy", {}).get("memory", {})
        room_names = [r["id"] for r in scene["rooms"]]
        max_entries = max(
            (sum(isinstance(v, dict) and v.get("room") == room for v in memory.values()) for room in room_names),
            default=0,
        )
        cache_height = max(340, 100 + max_entries * 57)
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {cache_height}" role="img" aria-label="Observed object cache">'
        ]
        for i, room in enumerate(room_names):
            x = 30 + i * 870 / len(room_names)
            width = 840 / len(room_names)
            elements.append(
                f'<rect x="{x}" y="10" width="{width}" height="{cache_height - 30}" rx="12" fill="#eef5fa" stroke="#9eb3c5"/>'
            )
            elements.append(f'<text x="{x + 12}" y="38" font-size="19">{html.escape(room)}</text>')
            known = [v for v in memory.values() if isinstance(v, dict) and v.get("room") == room]
            for j, value in enumerate(known):
                y = 77 + j * 57
                elements.append(f'<circle cx="{x + 17}" cy="{y}" r="5" fill="#087f9c"/>')
                elements.append(f'<text x="{x + 30}" y="{y + 5}" font-size="15">{html.escape(value["label"])}</text>')
                elements.append(
                    f'<text x="{x + 30}" y="{y + 23}" font-size="12" fill="#64748b">{html.escape(value["observation_id"])}</text>'
                )
            if not known:
                elements.append(
                    f'<text x="{x + 12}" y="80" font-size="13" fill="#64748b">No object observations</text>'
                )
        row["memory_svg"] = "".join(elements) + "</svg>"
        row["images"] = {
            key: "data:image/png;base64," + base64.b64encode((root / path).read_bytes()).decode()
            for key, path in event["images"].items()
        }
        rows.append(row)
    # Escaping '<' prevents even adversarial model/tool text from ending a script.
    data = json.dumps({"manifest": manifest, "events": rows, "metrics": metrics}).replace("<", "\\u003c")
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent task evidence</title>
<style>body{margin:0;background:#eef2f6;color:#172033;font:16px system-ui}main{max-width:1400px;margin:auto;padding:28px}
h1{font-size:30px;margin:8px 0}h2{font-size:18px;margin-top:0}.muted{color:#526477}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
section,.bar{background:white;border:1px solid #d5dee7;border-radius:12px;padding:18px;margin-bottom:18px}img{width:100%;border-radius:6px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;max-height:340px;overflow:auto}.controls{display:flex;gap:15px;align-items:center}input{flex:1}
button{padding:8px 18px;border:1px solid #64748b;background:white;border-radius:6px;cursor:pointer}svg{width:100%}.tag{display:inline-block;padding:6px 10px;background:#e0f2fe;border-radius:5px;margin:4px}
.missing{padding:40px 10px;background:#f1f5f9;text-align:center;color:#64748b}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<main><div class="muted">DYNAGRAPH / BENCHMARK PREFLIGHT</div><h1 id="title"></h1><p id="instruction"></p>
<div class="bar"><span class="tag" id="result"></span><span class="tag" id="policy-mode"></span><span class="tag">Paid calls: 0</span><p id="assistance" class="muted"></p></div>
<section><div class="controls"><button id="play">Play</button><label for="step">Step</label><input id="step" type="range" min="0" value="0"><output id="time"></output></div></section>
<section><h2>Room layout and measured trajectory</h2><div id="map"></div></section>
<div class="grid"><section><h2>Robot head observation</h2><div id="head"></div></section><section><h2>Evaluator overhead view</h2><div id="overhead"></div></section>
<section><h2>Decision and tool outcome</h2><pre id="policy"></pre></section><section><h2>Independent subgoal score</h2><pre id="score"></pre></section>
<section><h2>Observed object cache at this step</h2><p class="muted">Only recorded object observations; this is not learned DynaGraph memory.</p><div id="memory-map"></div><pre id="memory"></pre></section><section><h2>Exact model input references</h2><pre id="inputs"></pre></section></div>
<section><h2>Run provenance and limits</h2><pre id="manifest"></pre></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent), es=d.events, slider=document.getElementById('step');
const text=(id,v)=>document.getElementById(id).textContent=typeof v==='string'?v:JSON.stringify(v,null,2);
text('title',d.manifest.episode.id.replaceAll('_',' '));text('instruction',d.manifest.episode.instruction);
text('result',d.metrics.status+' · '+d.metrics.completed+'/'+d.metrics.total+' placement goals');
text('policy-mode',d.manifest.control==='local_agent'?'Local model · assisted skills':'Assisted control · no learned policy');text('assistance',d.manifest.assistance);text('manifest',d.manifest);slider.max=Math.max(0,es.length-1);
function showImage(id,e){const el=document.getElementById(id);el.replaceChildren();if(e.images[id]){const img=document.createElement('img');img.src=e.images[id];img.alt=id+' at '+e.observation_id;el.appendChild(img);}else{const p=document.createElement('div');p.className='missing';p.textContent='No image recorded at this event. Select an observation/tool boundary.';el.appendChild(p);}}
function show(){const e=es[+slider.value];if(!e)return;text('time',e.step+' · '+e.elapsed_s.toFixed(1)+' s · '+e.kind);document.getElementById('map').innerHTML=e.map_svg;
showImage('head',e);showImage('overhead',e);text('policy',e.policy);text('score',e.evaluator.score||'No measured score at this event');
text('memory',e.policy.memory??'No observation cache recorded at this event. Evaluator object positions are not agent beliefs.');
document.getElementById('memory-map').innerHTML=e.memory_svg;
text('inputs',e.kind==='model_input'?{text:e.policy.text,image_observation_ids:e.policy.model_input_observation_ids,has_image:e.policy.has_image}:'No model was called at this event.');}
slider.oninput=show;let timer=null;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;text('play','Play');return;}text('play','Pause');timer=setInterval(()=>{slider.value=(+slider.value+1)%es.length;show();},350);};show();
</script></html>"""
    (root / "report.html").write_text(template.replace("__DATA__", data))


def export_figures(root: Path, manifest: dict, events: list, metrics: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    from .fixture import obstacles

    scene = manifest["suite"]["scene"]
    observed = [e for e in events if e.get("evaluator", {}).get("positions")]
    if not observed:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for ax, event, label in zip(axes, [observed[0], observed[-1]], ["Before", "After"], strict=False):
        for room in scene["rooms"]:
            x0, y0, x1, y1 = room["bounds"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#edf2f6", edgecolor="#bac6d0"))
            ax.text((x0 + x1) / 2, y1 - 0.45, room["id"].title(), ha="center", fontsize=12)
        for x0, y0, x1, y1 in obstacles(scene):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#667788"))
        trace = [e["evaluator"]["robot_xyt"] for e in observed if e["step"] <= event["step"]]
        ax.plot([p[0] for p in trace], [p[1] for p in trace], color="#a16207", lw=2, label="Executed route")
        for name, pos in event["evaluator"]["positions"].items():
            obj = scene["objects"][name]
            ax.scatter(
                *pos[:2], c=[obj["color"]], marker="o" if obj["movable"] else "s", s=80, edgecolors="white", zorder=5
            )
            lane = sum(
                other != name and scene["objects"][other]["movable"] and p[0] <= pos[0] and abs(p[0] - pos[0]) < 0.9
                for other, p in event["evaluator"]["positions"].items()
            )
            ax.annotate(
                obj["label"].replace(" ", "\n", 1),
                pos[:2],
                xytext=(0, 18 + lane * 27 if obj["movable"] else -28),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        x, y, _ = event["evaluator"]["robot_xyt"]
        ax.add_patch(Circle((x, y), scene["footprint_radius_m"], color="#0891b2", alpha=0.35))
        ax.set(
            xlim=(min(r["bounds"][0] for r in scene["rooms"]) - 0.3, max(r["bounds"][2] for r in scene["rooms"]) + 0.3),
            ylim=(-2.4, 2.3),
            aspect="equal",
            xlabel="World x (m)",
            ylabel="World y (m)",
        )
        ax.set_title(
            f"{label} · step {event['step']} · {event['evaluator']['score']['completed']}/{metrics['total']} goals",
            loc="left",
        )
    fig.suptitle(
        manifest["episode"]["id"].replace("_", " ") + " — assisted fixture; evaluator ground truth", fontsize=14
    )
    for ext in ("png", "pdf"):
        fig.savefig(root / f"overview.{ext}", dpi=200)
    plt.close(fig)
    selected = [e for e in observed if "head" in e["images"]]
    if selected:
        from PIL import Image

        indices = (
            sorted({round(i * (len(selected) - 1) / min(5, len(selected) - 1)) for i in range(min(6, len(selected)))})
            if len(selected) > 1
            else [0]
        )
        fig, axs = plt.subplots(
            1, len(indices), figsize=(3.6 * len(indices), 3), squeeze=False, constrained_layout=True
        )
        for ax, i in zip(axs[0], indices, strict=False):
            e = selected[i]
            ax.imshow(Image.open(root / e["images"]["head"]))
            ax.set_title(f"Step {e['step']} · {e['kind']}", fontsize=9)
            ax.axis("off")
        fig.savefig(root / "storyboard.png", dpi=180)
        plt.close(fig)


def export_rerun(root: Path, manifest: dict, events: list):
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    from PIL import Image

    rec = rr.new_recording("emet-agent-tasks", recording_id=manifest["episode_fingerprint"][:24])
    rec.save(str(root / "episode.rrd"))
    rec.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(origin="world", name="Evaluator geometry"), rrb.TextDocumentView(origin="task")
                ),
                rrb.Vertical(rrb.Spatial2DView(origin="camera/head"), rrb.Spatial2DView(origin="camera/overhead")),
                rrb.Vertical(rrb.TextDocumentView(origin="policy"), rrb.TextDocumentView(origin="memory")),
            )
        )
    )
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    from .fixture import obstacles

    boxes = obstacles(manifest["suite"]["scene"])
    rec.log(
        "world/fixture",
        rr.Boxes3D(
            centers=[[(a + c) / 2, (b + d) / 2, 0.15] for a, b, c, d in boxes],
            half_sizes=[[(c - a) / 2, (d - b) / 2, 0.15] for a, b, c, d in boxes],
            colors=[100, 120, 140],
        ),
        static=True,
    )
    path = []
    for e in events:
        rec.set_time_sequence("step", e["step"])
        rec.set_time_seconds("elapsed", e["elapsed_s"])
        pose = e.get("evaluator", {}).get("robot_xyt")
        if pose:
            path.append([pose[0], pose[1], 0.05])
            rec.log("world/trajectory", rr.LineStrips3D([path], colors=[180, 120, 10]))
        positions = e.get("evaluator", {}).get("positions", {})
        if positions:
            rec.log(
                "world/evaluator_objects", rr.Points3D(list(positions.values()), labels=list(positions), radii=0.05)
            )
        for key in ("head", "wrist", "overhead"):
            if key in e["images"]:
                rec.log(f"camera/{key}", rr.Image(np.asarray(Image.open(root / e["images"][key]))))
            else:
                rec.log(f"camera/{key}", rr.Clear(recursive=True))
        rec.log("task", rr.TextDocument(json.dumps(e.get("evaluator", {}).get("score", {}), indent=2)))
        rec.log("policy", rr.TextDocument(json.dumps(e["policy"], indent=2)))
        rec.log(
            "memory",
            rr.TextDocument(json.dumps(e["policy"].get("memory", "No learned memory in assisted control"), indent=2)),
        )
    rec.flush()
    # Finalize the file before checksumming it; flush alone leaves the sink open.
    rec.disconnect()


def export_run(root: Path) -> dict:
    manifest, events, metrics = load_run(root)
    poses = [e["evaluator"]["robot_xyt"] for e in events if e.get("evaluator", {}).get("robot_xyt")]
    metrics["path_length_m"] = sum(math.dist(a[:2], b[:2]) for a, b in zip(poses, poses[1:], strict=False))
    errors = []
    export_html(root, manifest, events, metrics)
    for name, operation in (("figures", export_figures), ("rerun", export_rerun)):
        try:
            if name == "figures":
                operation(root, manifest, events, metrics)
            else:
                operation(root, manifest, events)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        import numpy as np
        from PIL import Image, ImageDraw

        from emet.eval.episode_video import write_rgb_sequence_mp4

        frames = []
        for e in events:
            if "head" not in e["images"] or "overhead" not in e["images"]:
                continue
            canvas = Image.new("RGB", (1280, 410), "#172033")
            for i, key in enumerate(("head", "overhead")):
                canvas.paste(Image.open(root / e["images"][key]).resize((640, 360)), (640 * i, 0))
            ImageDraw.Draw(canvas).text(
                (15, 375),
                f"ASSISTED / HEAD + EVALUATOR OVERHEAD | step {e['step']} | {e['kind']} | {e['elapsed_s']:.1f}s",
                fill="white",
            )
            frames.append(np.asarray(canvas))
        if frames:
            write_rgb_sequence_mp4(frames, root / "episode.mp4", fps=2)
        else:
            errors.append("video: no paired camera observations recorded")
    except Exception as exc:
        errors.append(f"video: {type(exc).__name__}: {exc}")
    required = ["report.html", "overview.png", "overview.pdf", "storyboard.png", "episode.rrd", "episode.mp4"]
    missing = [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
    result = {
        "complete": not missing and not errors,
        "missing": missing,
        "errors": errors,
        "source_event_hash": metrics.get("event_hash"),
        "artifacts": required,
    }
    metrics["evidence_complete"] = result["complete"]
    write_json(root / "metrics.json", metrics)
    export_html(root, manifest, events, metrics)
    result["exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in required if (root / name).is_file()
    }
    write_json(root / "artifacts.json", result)
    return result
