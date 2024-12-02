from __future__ import annotations


from datetime import date

import rio

from .. import components as comps


class RootComponent(rio.Component):
    """
    This page will be used as the root component for the app. This means, that
    it will always be visible, regardless of which page is currently active.

    This makes it the perfect place to put components that should be visible on
    all pages, such as a navbar or a footer.

    Additionally, the root page will contain a `rio.PageView`. Page views don't
    have any appearance on their own, but they are used to display the content
    of the currently active page. Thus, we'll always see the navbar and footer,
    with the content of the current page in between.
    """

    value: date = date.today()
    today: date = date.today()

    def on_value_change(self, event: rio.DateChangeEvent):
        # This function will be called whenever the input's value
        # changes. We'll display the new value in addition to updating
        # our own attribute.
        self.value = event.value
        # print(f"You've selected: {self.value}")
        self.today = date.today()
        print(f"Today: {self.today}")

    def build(self) -> rio.Component:
        return rio.Column(
            # The navbar contains a `rio.Overlay`, so it will always be on top
            # of all other components.
            comps.Navbar(),
            # Add some empty space so the navbar doesn't cover the content.
            rio.Spacer(min_height=5),
            # The page view will display the content of the current page.
            rio.PageView(
                grow_y=True,
            ),
            comps.Footer(),
        )
