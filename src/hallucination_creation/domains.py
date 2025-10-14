"""Domain definitions used for hallucination dataset collection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
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


ALL_DOMAINS: List[DomainPrompt] = [
        DomainPrompt(
            name="Global Health",
            description="Contemporary public health insights, medical research, and health policy developments.",
            seed_questions=[
                "Which clinical trial readouts between 2019 and 2023 triggered changes to WHO or CDC treatment guidelines?",
                "How did vaccine equity initiatives allocate doses across regions prior to 2024?",
                "What surveillance programs or financing instruments strengthened emerging disease detection before 2024?",
            ],
        ),
        DomainPrompt(
            name="Climate Policy and Environment",
            description="Environmental science, mitigation strategies, and long-horizon sustainability planning.",
            seed_questions=[
                "How did nationally determined contributions evolve between COP24 and COP28?",
                "Which jurisdictions implemented resilient infrastructure standards before 2024?",
                "What long-term biodiversity protections were funded through multilateral agreements prior to 2024?",
            ],
        ),
        DomainPrompt(
            name="Geopolitics and Security",
            description="International relations, treaty compliance, and shifting security alliances.",
            seed_questions=[
                "How did defense cooperation between Indo-Pacific partners change from 2019 to 2023?",
                "Which sanctions regimes reshaped critical exports before 2024?",
                "What diplomatic initiatives affected energy corridors or maritime access pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Technology Governance and AI",
            description="Policy, standards, and industry shifts surrounding advanced computing systems.",
            seed_questions=[
                "How did model licensing or export controls evolve for advanced semiconductors pre-2024?",
                "Which governments issued AI accountability frameworks between 2021 and 2023?",
                "What collaborative research programs linked academia and industry for trustworthy AI before 2024?",
            ],
        ),
        DomainPrompt(
            name="Finance and Markets",
            description="Macroeconomic indicators, capital flows, and structural shifts in financial systems.",
            seed_questions=[
                "Which central banks altered quantitative easing or tightening between 2020 and 2023, and why?",
                "How did sovereign debt restructurings progress in emerging markets prior to 2024?",
                "What cross-border payment initiatives or CBDC pilots advanced before 2024?",
            ],
        ),
        DomainPrompt(
            name="Culture and Media",
            description="Cultural production, media consolidation, and global content trends.",
            seed_questions=[
                "Which streaming mergers or licensing disputes reshaped audience access pre-2024?",
                "How did cultural institutions adapt business models after the 2020–2022 shutdowns?",
                "What international awards or festivals highlighted regional storytelling before 2024?",
            ],
        ),
        DomainPrompt(
            name="Cybersecurity and Data Privacy",
            description="Digital threat landscapes, regulation, and incident response coordination.",
            seed_questions=[
                "How did supply-chain attacks influence zero-trust adoption before 2024?",
                "Which data protection rulings reshaped cross-border data transfers between 2019 and 2023?",
                "What public-private partnerships funded national SOC capabilities pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Energy Transition and Grid Modernization",
            description="Clean energy deployment, grid reliability, and decarbonization finance.",
            seed_questions=[
                "What policies accelerated offshore wind or solar-plus-storage before 2024?",
                "How did grid operators integrate demand response between 2020 and 2023?",
                "Which countries issued green bonds earmarked for retiring coal assets pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Supply Chains and Manufacturing Resilience",
            description="Logistics realignment, reshoring strategies, and industrial capacity planning.",
            seed_questions=[
                "How did semiconductor supply agreements adjust between 2021 and 2023?",
                "Which trade corridors benefited from infrastructure upgrades before 2024?",
                "What manufacturing incentives diversified critical mineral sourcing pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Space Exploration and Satellite Systems",
            description="Orbital infrastructure, research missions, and commercial launch ecosystems.",
            seed_questions=[
                "Which lunar or Mars mission milestones were achieved between 2019 and 2023?",
                "How did satellite mega-constellations address collision mitigation before 2024?",
                "What international agreements governed spectrum or debris management pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Biotechnology and Genomics",
            description="Advances in gene editing, therapeutics, and biosecurity policy.",
            seed_questions=[
                "Which gene therapies gained regulatory approval between 2020 and 2023?",
                "How did global biosecurity frameworks evolve after the 2020 pandemic onset?",
                "What public-private consortia advanced mRNA or CRISPR research pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Education Policy and Learning Outcomes",
            description="K-12 and higher education reforms, digital access, and assessment changes.",
            seed_questions=[
                "Which countries invested in nationwide remote learning infrastructure before 2024?",
                "How did standardized testing policies shift between 2019 and 2023?",
                "What initiatives closed STEM achievement gaps prior to 2024?",
            ],
        ),
        DomainPrompt(
            name="Urban Development and Infrastructure",
            description="Housing, transit, and resilient city planning initiatives.",
            seed_questions=[
                "Which metropolitan areas enacted zoning overhauls to expand housing supply pre-2024?",
                "How did transit agencies finance electrification between 2020 and 2023?",
                "What smart-city pilots delivered measurable service improvements before 2024?",
            ],
        ),
        DomainPrompt(
            name="Agriculture and Food Security",
            description="Crop innovation, supply stability, and nutrition programs.",
            seed_questions=[
                "How did climate-resilient crop varieties perform in field trials before 2024?",
                "Which food aid reforms improved distribution logistics between 2019 and 2023?",
                "What trade policies stabilized staple grain markets pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Water Resources and Climate Resilience",
            description="Watershed management, drought planning, and equitable access initiatives.",
            seed_questions=[
                "Which regions adopted integrated water resource plans before 2024?",
                "How did desalination or reuse projects scale between 2019 and 2023?",
                "What transboundary agreements addressed river basin stress pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Human Rights and Social Justice",
            description="Legal reforms, civic movements, and accountability mechanisms.",
            seed_questions=[
                "Which truth and reconciliation processes issued findings before 2024?",
                "How did labor rights rulings affect gig or platform workers between 2019 and 2023?",
                "What corporate human rights benchmarks were adopted pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Macroeconomic Policy and Inflation Management",
            description="Fiscal interventions, price stability measures, and labor market dynamics.",
            seed_questions=[
                "How did fiscal stimulus unwinds proceed between 2021 and 2023?",
                "Which wage or price controls were trialed to address inflation pre-2024?",
                "What indicators signaled soft-landing prospects before 2024?",
            ],
        ),
        DomainPrompt(
            name="Transportation and Mobility Innovation",
            description="Mobility services, electrification, and logistics modernization.",
            seed_questions=[
                "How did EV charging infrastructure deployment progress before 2024?",
                "Which cities piloted congestion pricing or low-emission zones between 2019 and 2023?",
                "What autonomous freight corridors advanced regulatory approvals pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Disaster Preparedness and Climate Adaptation",
            description="Emergency management, early warning systems, and resilient recovery planning.",
            seed_questions=[
                "Which nations deployed multi-hazard early warning coverage before 2024?",
                "How did insurance or risk pools evolve to cover climate disasters between 2019 and 2023?",
                "What reconstruction frameworks prioritized resilient housing pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Science Policy and Research Funding",
            description="R&D prioritization, open science initiatives, and international collaboration.",
            seed_questions=[
                "How did major science funding agencies reallocate budgets between 2020 and 2023?",
                "Which open-access mandates reshaped publishing practices before 2024?",
                "What multinational research consortia delivered breakthrough findings pre-2024?",
            ],
        ),
        DomainPrompt(
            name="Legal and Regulatory Technology",
            description="Digital governance, antitrust enforcement, and platform accountability.",
            seed_questions=[
                "Which jurisdictions enacted AI or algorithmic transparency laws before 2024?",
                "How did antitrust cases against major platforms progress between 2019 and 2023?",
                "What fintech or crypto regulations changed institutional compliance pre-2024?",
            ],
        ),
    ]


def default_domains(limit: int | None = None) -> Iterable[DomainPrompt]:
    """Return a curated set of compelling domains for claim generation."""

    return list(islice(ALL_DOMAINS, limit)) if limit else list(ALL_DOMAINS)
