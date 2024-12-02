# from re import M
# import rio
# from pathlib import Path
# from typing import List
# from dataclasses import dataclass, field
# from datetime import datetime

# @dataclass
# class Project:
#     id: str
#     title: str
#     tech: str
#     description: str
#     link: str
#     image: str

# def default_projects() -> List[Project]:
#     return [
#         Project(
#             id="portfolio",
#             title="Portfolio Website",
#             tech="Built with Python using Rio-UI framework",
#             description="A personal portfolio website showcasing my work as a developer, research experience, and completed projects.",
#             link="https://github.com/linhduongtuan/portfolio_web",
#             image="assets/avatar_1.png"
#         ),
#         Project(
#             id="bioimage",
#             title="Biomedical Image Analysis",
#             tech="Built with Python using PyTorch and TensorFlow",
#             description="Deep learning models for biomedical image analysis, including cell segmentation and classification.",
#             link="https://github.com/linhduongtuan/bioimage-analysis",
#             image="assets/avatar.png"
#         )
#     ]

# @dataclass
# class HomeState:
#     current_project_index: int = 0
#     projects: List[Project] = field(default_factory=default_projects)

#     async def next_project(self):
#         self.current_project_index = (self.current_project_index + 1) % len(self.projects)

#     async def prev_project(self):
#         self.current_project_index = (self.current_project_index - 1) % len(self.projects)
# @rio.page(
#     name="Home",
#     url_segment="home-page",
# )
# class HomePage(rio.Component):
#     class State(HomeState): pass

#     def build(self) -> rio.Component:
#         return rio.Column(
#             # Header
#             rio.Container(
#                 rio.Column(
#                     rio.Row(
#                         rio.Column(
#                             rio.Text("My name is Linh Duong", style="heading1",
#                                      align_x=0.5),
#                             rio.Row(
#                                 rio.Text("I want to be like a ", style="heading3"),
#                                 rio.Link(rio.Text("Dolphin", style="text"), target_url="/about-page"),
#                                 rio.Text(" who learned to code.", style="heading3"),
#                             # grow_x=False,
#                             align_x=0.5,
#                             ),
#                         align_x=0.5,
#                         ),
#                         rio.Image(
#                             image=Path("assets/avatar.png"),
#                             min_width=10,
#                             min_height=10,
#                             corner_radius=2,
#                         ),
#                    ),
#                    rio.Spacer(),

#                     # Contact Info
#                     rio.Column(
#                         rio.Card(
#                             rio.Row(
#                                 rio.Text("Email:", style="heading3", align_x=0.5),
#                                 rio.Text("linhduongtuan@gmail.com", style="heading2", align_x=0.5),
#                                 rio.Button(rio.Icon("material/content-copy", min_width=0.01,
#                                                     min_height=0.01,
#                                                     margin=0.5,
#                                                     margin_x=0.5,
#                                                     margin_y=0.1,
#                                                     margin_right=0.1,
#                                                     margin_left=0.1,
#                                                     ),
#                                            on_press=lambda: print("Copied to clipboard!"),
#                                            margin=0.5,
#                                            ),
#                             ),

#                         margin=0.5,
#                         margin_x=0.5,
#                         margin_y=0.5,
#                         min_width=1,
#                         margin_right=0.1,
#                         margin_left=0.1,
#                         ),
#                         rio.Spacer(),
#                         rio.Card(
#                             rio.Column(
#                                 rio.Text("Please find my Curriculum Vitae attached", style="heading2", align_x=0.5, margin=0.5),
#                                 rio.Link(
#                                     rio.Button("Download CV", icon="material/download"),
#                                     target_url="assets/Linh_Duong_CV_20241111.pdf",
#                                     open_in_new_tab=False,
#                                 ),
#                             ),

#                         ),

#                     ),
#                 ),

#             ),
#             rio.Spacer(min_height=3),
#             # Projects
#             rio.Container(
#                 rio.Column(
#                     rio.Text("Projects and Research", style="heading1", align_x=0.5),
#                     rio.Spacer(min_height=1),
#                     rio.Card(
#                         rio.Column(
#                             rio.Row(
#                                 rio.Image(
#                                     image=Path("assets/avatar_1.png",
#                                                 ),
#                                                 min_width=10,
#                             min_height=90,
#                             corner_radius=2,
#                             margin=0.5,

#                                 ),
#                                 rio.Column(
#                                     rio.Text("XXXX", style="heading2"),
#                                     rio.Text("YYYYY", style="heading3"),
#                                     rio.Text("ZZZZZ", style="heading3"),
#                                     rio.Link(
#                                         rio.Button("View Project", icon="material/arrow-right", margin=0.5, min_width=0.01),
#                                         target_url="/research-page",
#                                         open_in_new_tab=False,
#                                     ),

#                                 ),

#                             ),


#                             rio.Row(
#                                 rio.Button(

#                                     rio.Icon("material/chevron-left"),
#                                     # on_press=self.State.prev_project,
#                                     on_press=lambda: print("Button pressed!"),

#                                 ),

#                                 rio.Button(

#                                     rio.Icon("material/chevron-right"),
#                                     # on_press=self.State.next_project,
#                                     on_press=lambda: print("Button pressed!"),


#                                 ),

#                              ),
#                         ),

#                     ),
#                     rio.Link(
#                         rio.Button("View All Projects"),
#                         target_url="/research-page",

#                     ),

#                 ),

#             ),

#         )


from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import rio


@dataclass
class Project:
    id: str
    title: str
    tech: str
    description: str
    link: str
    image: str


@dataclass
class HomeState:
    current_project_index: int = 0
    projects: List[Project] = field(
        default_factory=lambda: [
            Project(
                id="portfolio",
                title="Portfolio Website",
                tech="Built with Python using Rio-UI framework",
                description="A personal portfolio website showcasing my work as a developer, research experience, and completed projects.",
                link="https://github.com/linhduongtuan/portfolio_web",
                image="assets/avatar_1.png",
            ),
            Project(
                id="bioimage",
                title="Biomedical Image Analysis",
                tech="Built with Python using PyTorch and TensorFlow",
                description="Deep learning models for biomedical image analysis, including cell segmentation and classification.",
                link="https://github.com/linhduongtuan/bioimage-analysis",
                image="assets/avatar.png",
            ),
        ]
    )


@rio.page(name="Home", url_segment="home-page")
class HomePage(rio.Component):
    def __post_init__(self) -> None:
        super().__init__()
        self.state = HomeState()
        self._theme = {
            "spacing": {"xs": 0.2, "sm": 0.5, "md": 1, "lg": 1.5, "xl": 2},
            "margin": {"xs": 0.5, "sm": 1, "md": 1.5, "lg": 2},
            "radius": {"sm": 0.5, "md": 1.5},
        }

    def _create_header(self) -> rio.Component:
        return rio.Container(
            rio.Column(
                rio.Row(
                    rio.Column(
                        rio.Text("My name is Linh Duong", style="heading1"),
                        rio.Row(
                            rio.Text("I want to be like a ", style="heading3"),
                            rio.Link(
                                rio.Text("Dolphin", style="text"),
                                target_url="/about-page",
                            ),
                            rio.Text(" who learned to code.", style="heading3"),
                            align_x=0.5,
                        ),
                        align_x=0.5,
                    ),
                    rio.Image(
                        image=Path("personal_blog/assets/avatar.png"),
                        min_width=20,
                        min_height=20,
                        corner_radius=self._theme["radius"]["sm"],
                    ),
                ),
                spacing=self._theme["spacing"]["md"],
            )
        )

    def _create_contact_section(self) -> rio.Component:
        return rio.Column(
            rio.Card(
                rio.Row(
                    rio.Text("Email:", style="heading3", align_x=0.2),
                    rio.Text("linhduongtuan@gmail.com", style="heading2"),
                    rio.Button(
                        rio.Icon(
                            "material/content-copy", margin_left=0.5, margin_right=0.5
                        ),
                        on_press=lambda: print("Copied to clipboard!"),
                        margin=self._theme["margin"]["sm"],
                    ),
                    spacing=self._theme["spacing"]["sm"],
                ),
                margin=self._theme["margin"]["md"],
            ),
            rio.Card(
                rio.Column(
                    rio.Text(
                        "Please find my Curriculum Vitae attached",
                        style="heading2",
                        align_x=0.5,
                    ),
                    rio.Link(
                        rio.Button("Download CV", icon="material/download"),
                        target_url="assets/Linh_Duong_CV_20241111.pdf",
                    ),
                    spacing=self._theme["spacing"]["sm"],
                ),
            ),
            spacing=self._theme["spacing"]["md"],
        )

    def _create_project_card(self, project: Project) -> rio.Component:
        return rio.Card(
            rio.Column(
                rio.Row(
                    rio.Image(
                        image=Path(project.image),
                        min_width=10,
                        min_height=90,
                        corner_radius=self._theme["radius"]["sm"],
                        margin=self._theme["margin"]["xs"],
                    ),
                    rio.Column(
                        # rio.Link(
                        #     rio.Button(
                        #         "View Project",
                        #         icon="material/arrow-right",
                        #         margin=self._theme["margin"]["xs"],
                        #         min_width=0,
                        #         min_height=0,
                        #     ),
                        #     target_url=project.link,
                        #     open_in_new_tab=True,
                        # ),
                        rio.Text(project.title, style="heading2"),
                        rio.Text(project.tech, style="heading3"),
                        rio.Text(project.description, style="heading3"),
                        spacing=self._theme["spacing"]["sm"],
                    ),
                ),
                rio.Row(
                    rio.Button(
                        rio.Icon("material/chevron-left"),
                        on_press=self._prev_project,
                    ),
                    rio.Button(
                        rio.Icon("material/chevron-right"),
                        on_press=self._next_project,
                    ),
                    spacing=self._theme["spacing"]["sm"],
                ),
                spacing=self._theme["spacing"]["md"],
            )
        )

    async def _next_project(self):
        self.state.current_project_index = (self.state.current_project_index + 1) % len(
            self.state.projects
        )

    async def _prev_project(self):
        self.state.current_project_index = (self.state.current_project_index - 1) % len(
            self.state.projects
        )

    def build(self) -> rio.Component:
        # current_project = self.state.projects[self.state.current_project_index]

        return rio.Column(
            self._create_header(),
            rio.Spacer(min_height=1),
            self._create_contact_section(),
            rio.Spacer(min_height=1),
            rio.Container(
                rio.Column(
                    rio.Text("Projects and Research", style="heading1", align_x=0.5),
                    rio.Spacer(min_height=1),
                    # self._create_project_card(current_project),
                    rio.Link(
                        rio.Button("View All Projects"),
                        target_url="/research-page",
                    ),
                    spacing=self._theme["spacing"]["xs"],
                )
            ),
            spacing=self._theme["spacing"]["xs"],
            margin=self._theme["margin"]["xs"],
        )
