# DeepSeek compressor recurrent-state contract (2026-08-19)

## Contract

The DeepSeek learned compressor's complete recurrent buffers use the same ring
image produced by token-wise recurrence:

- ratio 4 retains the preceding completed group in the overlap half, writes the
  unfinished current group into the current-half prefix, and retains the last
  completed group in its inactive current-half tail;
- ratio 128 writes the unfinished group into the current-prefix and retains the
  last completed group in inactive tail slots;
- before any group completes, untouched slots are `0.0` in `kv_state` and
  negative infinity in `score_state`.

One-shot prefill, aligned multi-block append, and repeated incremental append
must therefore leave byte-identical recurrent state and compressed cache for
the same input sequence. This is stronger than output-only parity and prevents
transaction boundaries from changing observable cache snapshots.

## Root cause and fix

An exact DeepSeek-V4 downstream test compared a 2,048-row one-shot prefill with
two 1,024-row appends. Returned hidden state, logits, greedy tokens, and the
next decode were bit-identical, but the first ratio-128 layer differed at byte
16,384 of `main_kv_state`. The aligned-append path copied the last completed
group in inactive ring slots, matching token-wise recurrence, while one-shot
prefill zero-filled those slots. Those bytes were dormant, but their different
representations made complete recurrent state depend on transaction boundaries.

Aligned and incremental append already produced the token-wise ring image. The
Metal kernel's one-shot initializer now fills its inactive tail from the last
completed group before overlaying any current remainder. The focused test
compares one-shot, aligned batched append, and token-wise append state and cache
bit-for-bit for both production overlap dimensions and ratio 128, including
exact boundaries and partial remainders.

This is a correctness change, not a performance claim. Downstream hf2q must
consume a published crate containing the fix before relying on exact cache
state across cooperative transaction boundaries.
