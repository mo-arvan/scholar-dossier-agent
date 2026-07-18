from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

class Status(str, Enum):
    UNVERIFIED = "unverified"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"

class ConfidenceScore(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class DossierStatus(str, Enum):
    NEW = "new"
    NEEDS_REVIEW = "needs_review"
    PARTIALLY_VERIFIED = "partially_verified"
    FULLY_VERIFIED = "fully_verified"

class BaseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    confidence: ConfidenceScore
    status: Status = Status.UNVERIFIED

    source_flag: str = ""

class ProfileCandidate(BaseCandidate):
    url: str

class MediaCandidate(BaseCandidate):
    title: str
    url: str
    source: str
    year: Optional[int] = None
    snippet: str

class Position(BaseModel):
    """A structured representation of a single professional role using only basic types."""

    model_config = ConfigDict(extra="forbid")

    institution: str
    title: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    url: Optional[str] = None

class PositionCandidate(BaseCandidate):
    """A position in the career timeline, wrapping the simplified Position model."""

    data: Position

class Grant(BaseModel):
    """Structured representation of a grant/award record using basic types only."""

    model_config = ConfigDict(extra="forbid")

    pi_net_id: Optional[str] = None
    pi_name: Optional[str] = None

    title: Optional[str] = None
    sponsor: Optional[str] = None
    award_number: Optional[str] = None
    proposal_number: Optional[str] = None
    nih_project_num: Optional[str] = None
    nih_project_detail_url: Optional[str] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    amount: Optional[float] = None
    currency: Optional[str] = None

    status_label: Optional[str] = None
    role: Optional[str] = None
    institution: Optional[str] = None
    url: Optional[str] = None

class GrantCandidate(BaseCandidate):
    """A grant record wrapped with reasoning, confidence, and verification status."""

    data: Grant

class Publication(BaseModel):
    """A single publication record from OpenAlex / PubMed / ORCID (basic types only)."""

    model_config = ConfigDict(extra="forbid")

    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    openalex_id: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None

class PublicationCandidate(BaseCandidate):
    """A publication record wrapped with reasoning, confidence, and verification status."""

    data: Publication

class KeyProfiles(BaseModel):
    model_config = ConfigDict(extra="forbid")
    institutional_homepage: List[ProfileCandidate] = Field(default_factory=list)
    google_scholar: List[ProfileCandidate] = Field(default_factory=list)
    linkedin: List[ProfileCandidate] = Field(default_factory=list)
    orcid: List[ProfileCandidate] = Field(default_factory=list)
    personal_website: List[ProfileCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _drop_empty_profiles(self):

        for fname in type(self).model_fields:
            lst = getattr(self, fname)
            if isinstance(lst, list):
                setattr(self, fname, [p for p in lst if getattr(p, "url", "")])
        return self

class ClinicalImpactType(str, Enum):
    GUIDELINE = "guideline"
    DIAGNOSTIC_PROCEDURE = "diagnostic_procedure"
    THERAPEUTIC_PROCEDURE = "therapeutic_procedure"
    CLINICAL_TRIAL = "clinical_trial"
    DRUG = "drug"
    BIOMEDICAL_TECHNOLOGY = "biomedical_technology"
    SOFTWARE = "software"

class EconomicImpactType(str, Enum):
    PATENT = "patent"
    LICENSE = "license"
    COMMERCIAL_ENTITY = "commercial_entity"
    NON_PROFIT_ENTITY = "non-profit_entity"
    COST_SAVINGS_STUDY = "cost_savings_study"

class PolicyImpactType(str, Enum):
    COMMITTEE_PARTICIPATION = "committee_participation"
    EXPERT_TESTIMONY = "expert_testimony"
    POLICY_CITATION = "policy_citation"
    LEGISLATION = "legislation"
    STANDARD_DEVELOPMENT = "standard_development"

class CommunityImpactType(str, Enum):
    PUBLIC_HEALTH_INTERVENTION = "public_health_intervention"
    HEALTH_EDUCATION_RESOURCE = "health_education_resource"
    COMMUNITY_HEALTH_SERVICE = "community_health_service"
    HEALTH_SYSTEM_IMPROVEMENT = "health_system_improvement"
    CONSUMER_HEALTH_TOOL = "consumer_health_tool"

class ClinicalImpactCandidate(BaseCandidate):
    impact_type: ClinicalImpactType
    name: str
    url: Optional[str] = None
    year: Optional[int] = None
    identifier: Optional[str] = None
    summary: str

class EconomicImpactCandidate(BaseCandidate):
    impact_type: EconomicImpactType
    name: str
    url: Optional[str] = None
    year: Optional[int] = None
    identifier: Optional[str] = None
    summary: str

class PolicyImpactCandidate(BaseCandidate):
    impact_type: PolicyImpactType
    body_name: str
    role: Optional[str] = None
    document_name: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    summary: str

class CommunityImpactCandidate(BaseCandidate):
    """Represents a contribution to community or public health."""

    impact_type: CommunityImpactType
    name: str
    target_population: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    summary: str

class _AbsenceSection(BaseModel):
    """A section that can be explicitly marked empty via a model-set flag, distinguishing a
    confirmed absence from an empty/failed generation (only the latter is retried)."""

    model_config = ConfigDict(extra="forbid")
    confirmed_absent: bool = False

    confirmed_absent_reason: str = ""

    @model_validator(mode="after")
    def _absent_requires_empty(self):
        """confirmed_absent cannot hold alongside items; silently corrected rather than raised."""
        if getattr(self, "items", None) and self.confirmed_absent:
            self.confirmed_absent = False
        if not self.confirmed_absent:
            self.confirmed_absent_reason = ""
        return self

class MediaMentionsSection(_AbsenceSection):
    items: List[MediaCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _drop_empty(self):

        self.items = [m for m in self.items if (m.url or m.title or m.source)]
        return self

class _ImpactSection(_AbsenceSection):
    """Impact pillars share a drop-empty gate: an item with no name/body_name and no summary
    is a stray note, not an impact, so drop it."""

    @model_validator(mode="after")
    def _drop_empty_impacts(self):
        self.items = [
            it for it in self.items
            if (getattr(it, "name", "") or getattr(it, "body_name", "") or it.summary)
        ]
        return self

class ClinicalImpactsSection(_ImpactSection):
    items: List[ClinicalImpactCandidate] = Field(default_factory=list)

class CommunityImpactsSection(_ImpactSection):
    items: List[CommunityImpactCandidate] = Field(default_factory=list)

class EconomicImpactsSection(_ImpactSection):
    items: List[EconomicImpactCandidate] = Field(default_factory=list)

class PolicyImpactsSection(_ImpactSection):
    items: List[PolicyImpactCandidate] = Field(default_factory=list)

class GrantsSection(_AbsenceSection):
    items: List[GrantCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _drop_unidentified_grants(self):

        self.items = [
            g for g in self.items
            if (g.data.proposal_number or g.data.award_number
                or g.data.nih_project_num or g.data.title)
        ]
        return self

class Dossier(BaseModel):
    """The top-level dossier: identity fields plus all evidence sections."""

    model_config = ConfigDict(extra="forbid")

    net_id: str
    first_name: str
    last_name: str
    last_known_college: str
    last_known_department: str
    degree: str
    initial_research_title: str
    affiliation: str

    status: DossierStatus = DossierStatus.NEW
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    key_profiles: KeyProfiles = Field(default_factory=KeyProfiles)
    career_trajectory: List[PositionCandidate] = Field(default_factory=list)
    publications: List[PublicationCandidate] = Field(default_factory=list)
    grants: GrantsSection = Field(default_factory=GrantsSection)
    media_mentions: MediaMentionsSection = Field(default_factory=MediaMentionsSection)
    clinical_impacts: ClinicalImpactsSection = Field(default_factory=ClinicalImpactsSection)
    community_impacts: CommunityImpactsSection = Field(default_factory=CommunityImpactsSection)
    economic_impacts: EconomicImpactsSection = Field(default_factory=EconomicImpactsSection)
    policy_impacts: PolicyImpactsSection = Field(default_factory=PolicyImpactsSection)

    action_log: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _drop_empty_plain_lists(self):

        self.career_trajectory = [
            c for c in self.career_trajectory if (c.data.title or c.data.institution)
        ]
        self.publications = [
            p for p in self.publications if (p.data.title or p.data.doi or p.data.pmid)
        ]
        return self
