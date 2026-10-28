# Cached command-buffer reference-mode authority

`command_buffer_uses_unretained_references()` is a read-only public accessor
for the exact private cached decision already used by CommandEncoder creation
and rotation. Calling it first freezes that same decision, just as constructing
the first encoder did previously. Only MLX_UNRETAINED_REFS=1 enables unretained
references. It does not reread the environment after initialization, construct
a device, expose raw Metal objects, submit work, or change existing defaults.

A downstream retained-encoder experiment may use this authority to refuse
unretained mode before mutation. It still must retain each exact submitted
encoder, check every Result after a real fence, and preserve scratch ownership
and pool-reset boundaries. The accessor is not a health or physical-memory
claim. Unsupported/non-Metal execution remains unsupported: the host-only
accessor reports configuration, not successful encoder construction.

Host controls use fresh subprocesses for unset, 0, 1, true and 01, then change
and remove the variable and query from another thread. They invoke the actual
public accessor, not a mirrored parser or resettable substitute. No GPU/device
is used.

Release contract: this additive API is introduced in 0.15.12. Downstream must
wait for publication, registry verification and an exact version pin before
consuming it; a local path patch is not a released dependency.
