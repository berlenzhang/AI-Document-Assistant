from __future__ import annotations


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def escape(cell: object) -> str:
        return str(cell).replace("|", "\\|").replace("\n", " ")

    header_line = "| " + " | ".join(escape(h) for h in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(escape(c) for c in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])
