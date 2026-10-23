# Assessed evidence review follow-up

The assessed camera packet remains first in the final answer request. Explicit
FIND observations then occupy remaining image slots, followed by a detail crop
when one fits. Identical RGB is attached once. The configured image budget is
hard: a full packet can leave no slots for FIND or a crop. Count hints and room
context refer only to graph observations whose RGB is actually attached.

Packet requests skip ordinary graph image selection. Image roles and graph
observation indices describe the actual attachment positions; an unlocalized
context image does not gain a navigation target through its display index.

Assessed packets and pre-approach context live for the executor's episode rather
than expiring after 32 insertions. Memory therefore scales with episode captures;
these are not process-global caches. A future bounded cache must protect answer
evidence or explicitly report its loss, rather than silently changing the packet.

The depth gate deliberately requires close returns in 80% of all central pixels,
not 80% of only valid pixels. Missing depth remains unknown. Valid and invalid
fractions are reported independently; partial central depth is labeled as such.

Control profiles support lazy OmegaConf interpolation on access. They do not
support Hydra defaults composition and do not initialize or reset global Hydra.

The earlier paired smoke at 9f395d91 predates this repair and cannot qualify it.
Run fresh acceptance on the repaired revision before clearing PR #199.
