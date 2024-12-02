# personal_blog/pages/search_page.py
from __future__ import annotations
import rio
from urllib.parse import parse_qs, unquote


@rio.page(name="search", url_segment="search-page")
class SearchPage(rio.Component):
    """Search results page."""

    query: str = ""

    def __post_init__(self, context=None) -> None:
        super().__init__()
        self.context = context
        self.url_pattern = "/search-page"

    async def on_start(self) -> None:
        """Extract search query from URL parameters."""
        if self.context and self.context.request and self.context.request.url.query:
            query_params = parse_qs(self.context.request.url.query)
            # Decode the URL-encoded query parameter
            self.query = unquote(query_params.get("q", [""])[0])
        await self.force_refresh()

    def build(self) -> rio.Component:
        query_message = (
            f"Query: {self.query}" if self.query else "No search query provided"
        )
        return rio.Column(
            rio.Text("Search Results", style="heading1"),
            rio.Text(query_message),
            spacing=1,
        )
