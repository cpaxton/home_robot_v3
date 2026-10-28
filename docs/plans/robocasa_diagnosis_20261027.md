# RoboCasa search diagnosis

## Evidence

The corrected task is scored separately from the archived invalid jar/cab
protocol. Both main and candidate fail sugar-cube localization. Main localizes
a cabinet; candidate reaches the 1500 s episode cutoff during cabinet search.
No final candidate score is inferred from that incomplete run.

In the saved candidate object trace, early target distance is about 1.60 m;
after exploration it is about 4.04 m. Those are scorer-only distances, never
agent observations. Router and assessment RGBs repeatedly show brick walls.
The visual assessor's absence claims agree with those images.

Stationary probe `robocasa-stretch-optics-20261027` completes seven captures
with complete telemetry, upright base (up dot approximately 0.999995), level
camera, and valid RGB-D. A seeded top-down render shows the kitchen geometry;
the recorded spawn coordinates lie within the room bounds. An exterior spawn
is not supported by this evidence. The offline render initially exceeded the
model framebuffer size; rendering the saved XML at 640x480 succeeds.

Four-heading probe `robocasa-stretch-four-heading-20261027` is a separate
no-VLM, no-teleport diagnostic. Its first measured turn succeeds with about
0.012 m XY error and 0.099 rad yaw error, within the server's reported
tolerances. The resulting `view_01.png` clearly shows the stove. Remaining
turn results must be read from its final summary; one successful turn is not
a full control qualification.

Artifacts: `~/runs/emet/robocasa-stretch-optics-20261027/`,
`~/runs/emet/robocasa-stretch-four-heading-20261027/`, and
`~/runs/emet/robocasa-layout-diagnostic-20261027/`.

## Concrete context mismatches

1. The EQA tool describes `look_around` as a head/base scan. With the configured
   head sweep disabled on Stretch, it only captures the same heading. Calling
   it while facing a wall does not reveal the kitchen.
2. Frontier selection calls historical camera frames images of unexplored
   directions. They are associated context, not previews from the frontier,
   and do not certify visibility of a target there.

The small repair corrects both descriptions. It preserves configured scans,
motion controls, safety checks, budgets, grounding admission, and answer
authority. No automatic recovery/turn policy or target-specific rule is added.
Seventeen targeted room-policy/router/frontier tests pass. An existing fixture
now supplies an explicit identity planning transform instead of a MagicMock
that converts to an empty array.

## Interpretation and next check

Evidence points to a search/viewpoint problem, compounded by misleading tool
context and slow multi-chunk navigation. It does not establish that Qwen can
recognize/localize the small sugar cube once shown a clean view. The next live
S0 and RoboCasa candidate pilot must test the context-only repair before any
accuracy claim or stack merge. Do not increase range/budgets or alter scoring
to reclassify the existing failures.
