from __future__ import annotations

import rio


class NewsArticle(rio.Component):
    """
    Displays a news article with some visual separation from the background.
    Only shows the heading initially and expands on click to show full content.
    """

    markdown: str

    def _parse_heading_and_snippet(self, markdown: str) -> tuple[str, str]:
        lines = markdown.split("\n")
        heading = "No Heading Found"
        snippet_lines = []

        for i, line in enumerate(lines):
            if line.startswith("##"):
                heading = line.strip("#").strip()  # Remove ## and whitespace
            elif i < 2 and line.strip():
                snippet_lines.append(line.strip())

        snippet = " ".join(snippet_lines)
        return heading, snippet

    def build(self) -> rio.Component:
        heading, snippet = self._parse_heading_and_snippet(self.markdown)
        content = self.markdown

        return rio.Card(
            rio.Revealer(
                header=heading,
                content=rio.Column(
                    rio.Text(snippet, style="text", overflow="ellipsize"),
                    rio.Markdown(
                        content,
                        margin=0.5,
                        margin_x=0.5,
                        margin_y=0.5,
                        justify="justify",
                    ),
                    spacing=1,
                ),
                is_open=False,
            ),
            margin=1.5,
            margin_x=0.5,
            margin_y=0.5,
            margin_top=1,
            corner_radius=1,
        )
