"""Domain definitions used for hallucination dataset collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class DomainPrompt:
    """Represents a thematic area that guides claim generation."""

    name: str
    description: str
    seed_questions: List[str]

    def formatted_seed(self) -> str:
        """Return a short bullet list suitable for LLM prompting."""

        bullets = "\n".join(f"- {question}" for question in self.seed_questions)
        return f"{self.description}\nPotential angles:\n{bullets}"


def default_domains() -> Iterable[DomainPrompt]:
    """Return a curated set of compelling domains for claim generation."""

    return [
        DomainPrompt(
            name="Global Health",
            description="Contemporary public health insights, medical research, and health policy developments.",
            seed_questions=[
                "Which recent clinical trials have changed treatment standards?",
                "What global vaccination campaigns are underway this year?",
                "How are emerging diseases being monitored by the WHO?",
            ],
        ),
        DomainPrompt(
            name="Climate and Environment",
            description="Environmental science, climate policy, and planetary sustainability initiatives.",
            seed_questions=[
                "What climate accords were signed in the past two years?",
                "Which cities have adopted ambitious emissions targets recently?",
                "How are ecosystems responding to new conservation programs?",
            ],
        ),
        DomainPrompt(
            name="Geopolitics",
            description="International relations, treaties, conflicts, and diplomatic negotiations.",
            seed_questions=[
                "What alliances have shifted within the last 12 months?",
                "Which sanctions or trade agreements were enacted recently?",
                "How are regional conflicts affecting global supply chains?",
            ],
        ),
        DomainPrompt(
            name="Technology and AI",
            description="Latest advancements in artificial intelligence, cybersecurity, and software ecosystems.",
            seed_questions=[
                "What notable AI research papers were published this quarter?",
                "Which major data breaches were disclosed this year?",
                "How are governments regulating generative AI tools right now?",
            ],
        ),
        DomainPrompt(
            name="Finance and Markets",
            description="Macroeconomic indicators, market movements, and corporate developments.",
            seed_questions=[
                "Which central banks changed interest rates this quarter?",
                "What major IPOs or mergers closed recently?",
                "How are commodity prices reacting to geopolitical events?",
            ],
        ),
        DomainPrompt(
            name="Culture and Media",
            description="Film, literature, music, and cultural phenomena with significant public attention.",
            seed_questions=[
                "Which films have dominated recent awards seasons?",
                "What trends are emerging in global streaming platforms?",
                "How are major cultural institutions adapting to digital experiences?",
            ],
        ),
    ]
