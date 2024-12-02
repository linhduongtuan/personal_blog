from __future__ import annotations

from pathlib import Path
import rio


@rio.page(
    name="contact",
    url_segment="contact-page",
)
class Contact(rio.Component):
    """
    A sample page, containing a greeting and some testimonials.
    """

    def build(self) -> rio.Component:
        return rio.Column(
            rio.Image(
                Path("personal_blog/assets/avatar.png"),
                min_width=20,
                min_height=10,
                margin_right=2,
                align_x=0.2,
                corner_radius=1,
            ),
            rio.Markdown(
                """
## Contact me

- My Google Scholar: [Google Scholar Profile](https://scholar.google.com/citations?user=aZKRy1oAAAAJ&hl=en&authuser=2)
- My ResearchGate: [ResearchGate Profile](https://www.researchgate.net/profile/Linh-Duong-Tuan)
- My LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/linh-duong-746b0b9b/)
- My ORCID: [ORCID Profile](https://orcid.org/0000-0001-7411-1369)
- My GitHub: [GitHub Profile](https://github.com/linhduongtuan)
- And my email: [Email me](mailto:linhduongtuan@gmail.com)


            """,
                min_width=60,
                margin_bottom=20,
                align_x=0.5,
                align_y=0.5,
            ),
        )
