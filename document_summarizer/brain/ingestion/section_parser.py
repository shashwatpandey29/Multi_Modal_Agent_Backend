SECTION_HEADERS = [
    "abstract", "introduction", "related work",
    "method", "methodology", "experiments",
    "results", "discussion", "conclusion", "limitations"
]

def split_into_sections(text: str):
    sections = {}
    current = "unknown"
    sections[current] = []

    for line in text.splitlines():
        lower = line.lower().strip()
        if any(h in lower for h in SECTION_HEADERS):
            current = lower
            sections[current] = []
        sections[current].append(line)

    return {
        k: "\n".join(v)
        for k, v in sections.items()
        if len(v) > 5
    }
