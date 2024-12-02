from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel
import rio
from functools import lru_cache


class OutputType(Enum):
    PAPER = "paper"
    SOFTWARE = "software"
    AWARD = "award"
    PATENT = "patent"


class Output(BaseModel):
    type: OutputType
    title: str
    description: str
    url: Optional[str] = None
    year: int


class Research(BaseModel):
    title: str
    aims: str
    year_start: int
    year_end: int
    sponsor: str
    sponsor_url: str
    budget: str
    outputs: List[Output]


@rio.page(name="research", url_segment="research-page")
class ResearchPage(rio.Component):
    def __post_init__(self) -> None:
        super().__init__()
        self._theme = {
            "spacing": {"xs": 0.2, "sm": 0.5, "md": 1, "lg": 1.5, "xl": 2},
            "margin": {"xs": 0.5, "sm": 1, "md": 1.5, "lg": 2},
            "radius": {"sm": 0.5, "md": 1.5},
        }
        self._icons = {
            OutputType.PAPER: "📄",
            OutputType.SOFTWARE: "💻",
            OutputType.AWARD: "🏆",
            OutputType.PATENT: "📜",
        }

    def _create_output_title(self, output: Output) -> rio.Component:
        """Create title component for an output with appropriate icon and link."""
        title_text = f"{self._icons[output.type]} {output.title}"
        return (
            rio.Link(title_text, target_url=output.url, open_in_new_tab=True)
            if output.url
            else rio.Text(title_text)
        )

    def _create_output_description(self, output: Output) -> rio.Component:
        """Create description component for an output."""
        return rio.Text(
            f"{output.description} ({output.year})",
            style="text",
            margin_left=self._theme["margin"]["sm"],
        )

    def _create_output_section(self, outputs: List[Output]) -> rio.Component:
        """Create a section containing all outputs."""
        return rio.Column(
            *[
                rio.Column(
                    self._create_output_title(output),
                    self._create_output_description(output),
                    spacing=self._theme["spacing"]["xs"],
                )
                for output in outputs
            ],
            spacing=self._theme["spacing"]["sm"],
        )

    def _create_project_header(self, project: Research) -> rio.Component:
        """Create project header with title and years."""
        return rio.Text(
            f"{project.title} ({project.year_start}-{project.year_end})",
            style="heading3",
            margin_top=self._theme["margin"]["sm"],
            margin_x=self._theme["margin"]["lg"],
        )

    def _create_sponsor_link(self, project: Research) -> rio.Component:
        """Create sponsor link component."""
        return rio.Link(
            f"Sponsor: {project.sponsor} ({project.budget})",
            target_url=project.sponsor_url,
            open_in_new_tab=True,
            margin_x=self._theme["margin"]["lg"],
        )

    def _create_project_card(self, project: Research) -> rio.Component:
        """Create a card component for a research project."""
        return rio.Card(
            content=rio.Column(
                self._create_project_header(project),
                self._create_sponsor_link(project),
                rio.Revealer(
                    header="Project Aims",
                    content=rio.Markdown(
                        project.aims,
                        margin=self._theme["margin"]["xs"],
                        justify="justify",
                        margin_x=self._theme["margin"]["lg"],
                    ),
                    is_open=False,
                    margin_x=self._theme["margin"]["lg"],
                ),
                rio.Revealer(
                    header="Outputs",
                    content=self._create_output_section(project.outputs),
                    is_open=False,
                    margin_x=self._theme["margin"]["lg"],
                ),
                spacing=self._theme["spacing"]["sm"],
            ),
            margin=self._theme["margin"]["sm"],
            min_width=80,
            margin_x=self._theme["margin"]["lg"],
            margin_y=self._theme["margin"]["lg"],
            align_x=0.5,
            align_y=0.5,
            grow_x=True,
            corner_radius=self._theme["radius"]["md"],
        )

    @lru_cache(maxsize=1)
    def _group_projects_by_year(self) -> Dict[int, List[Research]]:
        """Group projects by start year with caching."""
        projects_by_year: Dict[int, List[Research]] = {}
        for project in RESEARCH_PROJECTS:
            if project.year_start not in projects_by_year:
                projects_by_year[project.year_start] = []
            projects_by_year[project.year_start].append(project)
        return projects_by_year

    def build(self) -> rio.Component:
        """Build the research page component."""
        projects_by_year = self._group_projects_by_year()

        return rio.Column(
            rio.Text("Research Projects", style="heading1"),
            *[
                rio.Column(
                    rio.Text(str(year), style="heading2"),
                    *[
                        self._create_project_card(project)
                        for project in projects_by_year[year]
                    ],
                    spacing=self._theme["spacing"]["sm"],
                )
                for year in sorted(projects_by_year.keys(), reverse=True)
            ],
            min_width=80,
            margin=self._theme["margin"]["lg"],
            align_x=0.5,
            align_y=0.5,
            grow_x=True,
            spacing=self._theme["spacing"]["lg"],
        )


RESEARCH_PROJECTS = [
    Research(
        title="AI-Powered Medical Image Analysis for Early Disease Detection",
        aims="Develop and validate deep learning models for automated detection of diseases from various medical imaging modalities, focusing on tuberculosis, COVID-19, and breast cancer screening.",
        year_start=2023,
        year_end=2025,
        sponsor="European Research Council",
        sponsor_url="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en",
        budget="€750,000",
        outputs=[
            Output(
                type=OutputType.PAPER,
                title="Detection of tuberculosis from chest X-ray images: Boosting performance with Vision Transformer",
                description="Published in Expert Systems with Applications (Impact Factor: 8.665)",
                url="https://doi.org/10.1016/j.eswa.2021.115519",
                year=2023,
            ),
            Output(
                type=OutputType.SOFTWARE,
                title="MedicalVisionAI",
                description="Open-source medical image analysis toolkit with 1000+ GitHub stars",
                url="https://github.com/medicalvisionai",
                year=2023,
            ),
            Output(
                type=OutputType.AWARD,
                title="Best Paper Award",
                description="MICCAI 2023 Conference - Medical Image Computing and Computer Assisted Intervention",
                year=2023,
            ),
        ],
    ),
    Research(
        title="Generative AI for Drug Discovery",
        aims="Leverage large language models and generative AI to accelerate drug discovery process by predicting protein structures and generating novel molecular compounds.",
        year_start=2022,
        year_end=2024,
        sponsor="Horizon Europe",
        sponsor_url="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en",
        budget="€1,200,000",
        outputs=[
            Output(
                type=OutputType.PAPER,
                title="ProteinGPT: Large Language Models for Protein Structure Prediction",
                description="Published in Nature Machine Intelligence",
                url="https://doi.org/example",
                year=2023,
            ),
            Output(
                type=OutputType.PATENT,
                title="Method for Generating Novel Molecular Compounds Using AI",
                description="Patent No. EP123456",
                url="https://patent.example",
                year=2023,
            ),
            Output(
                type=OutputType.SOFTWARE,
                title="MoleculeGen",
                description="AI-powered molecular generation platform",
                url="https://github.com/moleculegen",
                year=2023,
            ),
        ],
    ),
    Research(
        title="Multimodal Learning for Healthcare",
        aims="Develop innovative multimodal learning approaches combining medical imaging, clinical notes, and genomic data for personalized medicine.",
        year_start=2021,
        year_end=2024,
        sponsor="National Science Foundation",
        sponsor_url="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en",
        budget="€500,000",
        outputs=[
            Output(
                type=OutputType.PAPER,
                title="MultiHealth: A Multimodal Deep Learning Framework for Healthcare",
                description="Published in Nature Methods",
                url="https://doi.org/example2",
                year=2022,
            ),
            Output(
                type=OutputType.SOFTWARE,
                title="HealthFusion",
                description="Framework for multimodal medical data fusion",
                url="https://github.com/healthfusion",
                year=2022,
            ),
            Output(
                type=OutputType.AWARD,
                title="Innovation Award",
                description="European Healthcare Innovation Awards 2022",
                year=2022,
            ),
        ],
    ),
    Research(
        title="Federated Learning for Privacy-Preserving Healthcare AI",
        aims="Design and implement federated learning systems for training medical AI models across multiple institutions while preserving patient privacy.",
        year_start=2020,
        year_end=2023,
        sponsor="EU Privacy Tech Grant",
        sponsor_url="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en",
        budget="€650,000",
        outputs=[
            Output(
                type=OutputType.PAPER,
                title="SecureHealth: Privacy-Preserving Federated Learning in Healthcare",
                description="Published in IEEE Transactions on Medical Imaging",
                url="https://doi.org/example3",
                year=2022,
            ),
            Output(
                type=OutputType.SOFTWARE,
                title="FedHealth",
                description="Open-source federated learning framework for healthcare",
                url="https://github.com/fedhealth",
                year=2021,
            ),
            Output(
                type=OutputType.PATENT,
                title="System for Privacy-Preserving Medical AI Training",
                description="Patent No. US987654",
                url="https://patent.example2",
                year=2022,
            ),
        ],
    ),
    Research(
        title="Explainable AI for Clinical Decision Support",
        aims="Develop interpretable AI models that provide transparent and explainable recommendations for clinical decision support systems.",
        year_start=2019,
        year_end=2022,
        sponsor="Medical Research Council",
        sponsor_url="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en",
        budget="€450,000",
        outputs=[
            Output(
                type=OutputType.PAPER,
                title="XAI-Med: Explainable AI Framework for Medical Decision Support",
                description="Published in JAMA Network Open",
                url="https://doi.org/example4",
                year=2021,
            ),
            Output(
                type=OutputType.SOFTWARE,
                title="ExplainMed",
                description="Toolkit for medical AI model interpretation",
                url="https://github.com/explainmed",
                year=2020,
            ),
            Output(
                type=OutputType.AWARD,
                title="Clinical Impact Award",
                description="International Conference on Healthcare Informatics 2021",
                year=2021,
            ),
        ],
    ),
]
