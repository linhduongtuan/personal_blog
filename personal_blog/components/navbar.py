from __future__ import annotations
import rio
from urllib.parse import quote
from datetime import datetime, timezone
from personal_blog import data_models, persistence
from typing import Optional


class Navbar(rio.Component):
    """A responsive navigation bar component with search functionality."""

    def __post_init__(self) -> None:
        super().__init__()
        self.search_query = ""
        self._nav_links = [
            ("Home", "material/home", "home"),
            ("Research", "material/science", "research"),
            ("Publications", "material/library_books", "publications"),
            ("Blog", "material/article", "blog"),
            ("About me", "material/person", "about"),
            ("Contact", "material/person", "contact"),
            # ("Login", "material/login", "login"),
        ]

    @rio.event.on_page_change
    async def on_page_change(self) -> None:
        await self.force_refresh()

    async def _handle_search(self, query: str) -> None:
        """Handle search query submission."""
        if query.strip():
            encoded_query = quote(query.strip())
            await self.session.app.navigate(f"/search-page?q={encoded_query}")  # type: ignore

    async def handle_search_input(self, event: rio.TextInputChangeEvent) -> None:
        """Update search query as user types."""
        self.search_query = event.text
        await self.force_refresh()

    async def handle_search_submit(self, event: rio.TextInputConfirmEvent) -> None:
        """Handle search submission."""
        await self._handle_search(self.search_query)

    async def on_logout(self) -> None:
        """Handle user logout."""
        user_session = self.session[data_models.UserSession]
        pers = self.session[persistence.UserPersistence]

        await pers.update_session_duration(
            user_session,
            new_valid_until=datetime.now(tz=timezone.utc),
        )

        # Clear session data
        self.session.detach(data_models.AppUser)
        self.session.detach(data_models.UserSession)
        self.session.navigate_to("/")

    def _get_active_page(self) -> Optional[str]:
        """Get the current active page URL segment."""
        try:
            if self.session.active_page_instances:
                return self.session.active_page_instances[0].url_segment
        except (IndexError, AttributeError):
            return None
        return None

    def _build_search_box(self) -> rio.Component:
        """Build the search box component."""
        return rio.Row(
            rio.Icon(
                "material/search",
                fill="primary",
                min_width=2,
                align_x=0.9,
                margin=0.5,
            ),
            rio.TextInput(
                text=self.search_query,
                on_change=self.handle_search_input,
                on_confirm=self.handle_search_submit,
                prefix_text="Search...",
                style="pill",
                min_width=10,
            ),
            spacing=0,
            min_width=5,
            margin=0.5,
        )

    def _build_nav_button(
        self, label: str, icon: str, url: str, active_page: str
    ) -> rio.Component:
        """Build a navigation button."""
        return rio.Link(
            rio.Button(
                label,
                icon=icon,
                style="major" if active_page == f"{url}-page" else "plain-text",
            ),
            f"/{url}-page",
        )

    def _build_auth_buttons(self) -> rio.Component:
        """Build authentication-related buttons."""
        try:
            self.session[data_models.AppUser]
            return rio.Button(
                "Logout",
                icon="material/logout",
                style="plain-text",
                on_press=self.on_logout,
            )
        except KeyError:
            return rio.Link(
                rio.Button(
                    "Login",
                    icon="material/login",
                    style="plain-text",
                ),
                "/",
            )

    def build(self) -> rio.Component:
        """Build the navbar component."""
        active_page = self._get_active_page()

        nav_buttons = [
            self._build_nav_button(label, icon, url, active_page)  # type: ignore
            for label, icon, url in self._nav_links
        ]

        navbar_content = rio.Row(
            *nav_buttons,
            self._build_auth_buttons(),
            self._build_search_box(),
            spacing=2.5,
            align_y=10,
            align_x=0.5,
            min_height=1,
            min_width=30,
        )

        return rio.Overlay(
            rio.Rectangle(
                content=navbar_content,
                fill=self.session.theme.neutral_color,
                corner_radius=0.5,
                shadow_radius=0.5,
                shadow_color=self.session.theme.shadow_color,
                shadow_offset_y=0.1,
                align_y=0,
                margin_x=1.5,
                margin=0.5,
            )
        )
