# Spilled blobs

When a pasted artifact (log, source file, JSON, diff) is too large for the
conversation, the pre-pass pipeline spills the original to a content-addressed
blob store and replaces it in the message with a `[router-blob v2 <hash>]`
pointer plus a verbatim excerpt. The excerpt is code-sliced, not
model-written — what you see is real data, just partial.

Two tools work on that hash:

- `read_blob(hash, start_line, end_line)` — read back more of the original.
  Omit both line args for the whole thing (refused if too large — page
  through it, or use `grep_blob` instead). Line numbers are 1-indexed and
  inclusive.
- `grep_blob(hash, pattern, context_lines)` — regex search across the full
  blob, returning matches with surrounding context. Usually faster than
  paging through `read_blob` when you're looking for something specific (an
  error, a function name, a key).

The hash always comes from a `[router-blob v2 <hash>]` marker already visible
in the conversation — never guess or invent one.
