from __future__ import annotations
from pathlib import Path
from datetime import date, datetime
import pytz
import rio
import asyncio

# Define the timezone for strict UTC+1 (no daylight saving time)
utc_plus_1 = pytz.timezone("Etc/GMT-1")

# Get the current date and time in UTC+1
current_time_utc_plus_1 = datetime.now(utc_plus_1)

# Format the date as year:month:day
formatted_date = current_time_utc_plus_1.strftime("%Y-%m-%d-%H:%M:%S")


class Footer(rio.Component):
    """A simple, static component which displays a footer with the blog name and additional elements."""

    value: date = date.today()
    today: date = date.today()
    current_time: str = formatted_date
    _update_clock_task: asyncio.Task | None = None

    async def on_mount(self):
        if self._update_clock_task is None:
            self._update_clock_task = asyncio.create_task(self._update_clock())

    async def on_unmount(self):
        if self._update_clock_task:
            self._update_clock_task.cancel()
            self._update_clock_task = None

    async def _update_clock(self):
        try:
            while True:
                utc_plus_1 = pytz.timezone("Etc/GMT-1")
                current = datetime.now(utc_plus_1)
                self.current_time = current.strftime("%Y-%m-%d %H:%M:%S")
                await asyncio.sleep(1)  # Update every second
        except asyncio.CancelledError:
            pass

    def on_value_change(self, event: rio.DateChangeEvent):
        self.value = event.value
        self.today = date.today()
        print(f"Today: {self.today}")

    def build(self) -> rio.Component:
        return rio.Card(
            content=rio.Row(
                rio.Column(
                    rio.Link(
                        rio.Calendar(
                            value=self.value,
                            on_change=self.on_value_change,
                            margin=0,
                            margin_x=0,
                            margin_y=0,
                        ),
                        target_url="https://calendar.google.com/calendar/u/2/r/day?cid=Y19iNTc2Y2QxNjRkYjIwZTFhMTBkYWVmNDIzOWRlNjlhZDg0ZGY5YjhmZmQ0NWJmYjU5Yzc4MGNmMmI5NmMyMTA3QGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20&pli=1",
                        open_in_new_tab=True,
                    ),
                    spacing=0.1,
                    align_x=0.6,
                    align_y=0.5,
                    margin_y=0.1,
                    margin_top=2,
                ),
                # Blog Name and Social Media Links
                rio.Column(
                    rio.Row(
                        rio.Text(
                            "Linh Duong Info:", justify="center", style="heading3"
                        ),
                        rio.Link(
                            "Github",
                            target_url="https://github.com/linhduongtuan",
                            open_in_new_tab=True,
                        ),
                        rio.Text("|"),
                        rio.Link(
                            "Google Scholar",
                            target_url="https://scholar.google.com/citations?user=aZKRy1oAAAAJ&hl=en&authuser=2",
                            open_in_new_tab=True,
                        ),
                        rio.Text("|"),
                        rio.Link(
                            "ResearchGate",
                            target_url="https://www.researchgate.net/profile/Linh-Duong-Tuan",
                            open_in_new_tab=True,
                        ),
                        rio.Text("|"),
                        rio.Link(
                            "ORCID",
                            target_url="https://orcid.org/0000-0001-74119285",
                            open_in_new_tab=True,
                        ),  # Corrected ORCID link
                        rio.Text("|"),
                        rio.Link(
                            "LinkedIn",
                            target_url="https://www.linkedin.com/in/your-linkedin-profile",  # Replace with your LinkedIn profile
                            open_in_new_tab=True,
                        ),
                        spacing=0.25,
                        align_x=0.5,
                    ),
                    rio.Column(
                        rio.Image(
                            Path("personal_blog/assets/avatar.png"),
                            min_width=20,
                            min_height=10,
                            margin_right=2,
                            align_x=0.2,
                            corner_radius=1,
                        ),
                        # Copyright Information
                        rio.Text(
                            "© 2024 Linh Duong Blog. All rights reserved.",
                            justify="center",
                            style="dim",
                        ),
                        rio.Text(
                            f"Time now (UTC+1): {formatted_date}", justify="center"
                        ),
                        spacing=0.5,
                        margin=0.5,
                        align_x=0.5,
                    ),
                ),
            ),
            color="hud",
            corner_radius=5,
            margin=0,
            min_width=0,
            min_height=0,
            grow_y=True,
            elevate_on_hover=True,
            colorize_on_hover=False,
            margin_top=0,
            on_press=lambda: print("Card clicked!"),
            ripple=True,
        )
