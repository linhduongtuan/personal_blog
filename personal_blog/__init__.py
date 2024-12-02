from __future__ import annotations

from pathlib import Path

import rio

from . import components as comps
from . import persistence

# from .pages.search_page import SearchPage

# Define a theme for Rio to use.
#
# You can modify the colors here to adapt the appearance of your app or website.
# The most important parameters are listed, but more are available! You can find
# them all in the docs
#
# https://rio.dev/docs/api/theme
# theme = rio.Theme.from_colors(
#     primary_color=rio.Color.from_hex("01dffdff"),
#     secondary_color=rio.Color.from_hex("0083ffff"),
#     mode="light",
# )


async def init_app() -> rio.EventHandler:
    await persistence.UserPersistence.ensure_pool()


theme = rio.Theme.pair_from_colors(
    primary_color=rio.Color.from_hex("01dffdff"),
    secondary_color=rio.Color.from_hex("0083ffff"),
)

themes = rio.Theme.from_colors(
    mode="dark",
    corner_radius_large=3.48,
    corner_radius_medium=2.61,
    corner_radius_small=2.34,
)


# Create the Rio app
app = rio.App(
    name="multipage-website",

    build=comps.RootComponent,
    # theme=theme,
    theme=themes,
    assets_dir=Path(__file__).parent / "assets",
    # pages=SearchPage(),
)
