def trim_context(chunks, max_chars=3000):
    """
    Limit total context sent to LLM.
    This drastically improves speed.
    """
    text = ""
    for chunk in chunks:
        if len(text) + len(chunk) > max_chars:
            break
        text += chunk + "\n"
    return text
