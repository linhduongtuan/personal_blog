from __future__ import annotations
import pandas as pd
import plotly.express as px
from pathlib import Path
import rio


@rio.page(
    name="About me",
    url_segment="about-page",
)
class AboutPage(rio.Component):
    """
    A sample page, which displays a humorous description of the company.
    """

    metrics: pd.DataFrame = pd.DataFrame(
        {
            "Metric": ["Citations", "H-index", "i10-index"],
            "All Time": [433, 8, 8],
            "Since 2019": [422, 8, 8],
        }
    )
    yearly: pd.DataFrame = pd.DataFrame(
        {
            "Year": [2019, 2020, 2021, 2022, 2023, 2024],
            "Citations": [4, 7, 51, 84, 142, 148],
        }
    )

    async def update_plotly_fig(self):
        self.metrics["All Time"] = [433, 8, 8]
        self.metrics["Since 2019"] = [422, 8, 8]
        self.yearly["Citations"] = [4, 7, 51, 84, 142, 148]
        print(self.metrics)

    def build(self) -> rio.Component:
        fig = px.bar(
            self.yearly,
            x="Year",
            y="Citations",
            title="Citations per year",
            labels={"Citations": "Number of Citations"},
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Citations",
            showlegend=True,
        )
        return rio.Column(
            rio.Image(
                Path("personal_blog/assets/Linh_2540.jpg"),
                min_width=5,
                min_height=10,
                margin_right=2,
                align_x=0.2,
                corner_radius=1,
            ),
            rio.Markdown(
                """
## About me

- My name is Linh Duong. I am a Ph.D. in Biotechnology and a postdoc at KTH Royal Institute of Technology.

- **My research interests lie in:**
    
    \u2217 Research on bacterial virulence factors
    
    \u2217 Study cellular physiology

    \u2217 Investigate interactions with cellular signaling pathways
   
    \u2217 Elucidate the role of the microbiome in health and disease
     

-  **Additionally, I leverage computational biology to:**

    \u2217 Analyze genetic data

    \u2217 Develop predictive models

    \u2217 Study non-communicable diseases

    \u2217 Apply machine learning to biological data and medical images



            """,
                min_width=60,
                margin_bottom=4,
                align_x=0.5,
                align_y=0.5,
            ),
            rio.Column(
                rio.Text("Here is my number of citations yearly", style="heading2"),
                rio.Markdown(
                    "[Please click here to access my Google Scholar profile]((https://scholar.google.com/citations?user=aZKRy1oAAAAJ&hl=en&authuser=2))"
                ),
                rio.Plot(
                    figure=fig,
                    min_height=20,
                    min_width=10,
                    background=self.session.theme.neutral_color,
                ),
                spacing=2,
                margin=1,
                align_x=0.5,
                align_y=0,
            ),
        )
