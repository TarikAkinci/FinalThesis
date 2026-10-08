"""
Pure token-span helpers, split out of attention_probe.py so importing them
doesn't pull in inseq (attention_probe.py does `import inseq` at module level
for its own -- now legacy -- attribution workflow, but these two functions
never touch it). Anything that only needs span-finding, not the Inseq/
CONDITIONS/prepare_items machinery, should import from here instead.

attention_probe.py itself now imports these back from here, so existing
scripts that do `from attention_probe import find_token_span` are unaffected.
"""


def _reconstruct(tokens):
    """tokens: list of BPE token strings (Ġ/Ċ-marked). Returns (offsets, full)
    where offsets[i] is the character index at which token i starts in the
    concatenated string `full`."""
    cleaned = [t.replace("Ġ", " ").replace("Ċ", "\n") for t in tokens]
    offsets, full = [], ""
    for tok in cleaned:
        offsets.append(len(full))
        full += tok
    return offsets, full


def _char_span_to_token_span(offsets, idx, end_char):
    start_idx = max(next(i for i, off in enumerate(offsets) if off > idx) - 1, 0)
    end_idx = next((i for i, off in enumerate(offsets) if off >= end_char), len(offsets)) - 1
    return start_idx, end_idx


def find_token_span(tokens, target_text, last=False):
    """target_text: literal substring to locate. Returns inclusive
    (start_idx, end_idx) or None. last=True takes the LAST occurrence: use it
    for any span the template places after the item text, otherwise an item
    whose own text contains the string (e.g. "a man") captures the match.

    NOTE: only safe for FIXED strings we control (the instruction text, a
    demographic phrase). For arbitrary item text, use find_span_by_delimiters
    instead -- item text can itself contain quote characters that collide
    with PROMPT's own wrapping quotes (e.g. D3CODE item_id=1560 starts and
    ends with a literal '"'), which can make substring search land on the
    wrong occurrence or otherwise misbehave."""
    offsets, full = _reconstruct(tokens)
    idx = full.rfind(target_text) if last else full.find(target_text)
    if idx == -1:
        return None
    end_char = idx + len(target_text)
    return _char_span_to_token_span(offsets, idx, end_char)


def find_span_by_delimiters(tokens, prefix, suffix, search_from=0):
    """Locate a span by its FIXED surrounding delimiters instead of its own
    (arbitrary, possibly special-character-containing) content -- e.g. item
    text wrapped as 'Text: "{text}"\\n' in PROMPT. `prefix` is the literal
    text immediately before the span (e.g. 'Text: "'), `suffix` the literal
    text immediately after it. Both are always-constant parts of the prompt
    template, so this is robust to whatever the span's own content happens to
    contain. Returns inclusive (start_idx, end_idx) or None if either
    delimiter isn't found."""
    offsets, full = _reconstruct(tokens)
    prefix_idx = full.find(prefix, search_from)
    if prefix_idx == -1:
        return None
    start_char = prefix_idx + len(prefix)
    suffix_idx = full.find(suffix, start_char)
    if suffix_idx == -1:
        return None
    if suffix_idx <= start_char:
        return None
    return _char_span_to_token_span(offsets, start_char, suffix_idx)
