# FILE: src/query_builder.py
"""
PICO-lite PubMed Query Builder.

Converts free-text medical questions into high-recall, high-precision PubMed queries
using PICO decomposition, term normalization, MeSH injection, and proper Boolean assembly.

Example Usage:
    from src.query_builder import PubMedQueryBuilder, PICOQuery
    
    builder = PubMedQueryBuilder()
    
    # From structured input
    pico = PICOQuery(
        population=["melanoma", "cutaneous melanoma"],
        intervention=["treatment", "therapy"],
        modifiers=["stage II", "stage 2"]
    )
    query = builder.build_query(pico)
    
    # From free text (requires LLM decomposition)
    query = await builder.from_free_text("What is the latest treatment for stage II melanoma?")
"""
#Convert a messy human medical question into a precise, high-recall PubMed Boolean query.
import logging
import re
#from dataclasses import dataclass, field
#from enum import Enum
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PICO Query Model
# =============================================================================
#  BASE MODEL A strict gatekeeper that checks data before letting it in.
# so every pico query is checked and validated before it is used.
class PICOQuery(BaseModel): #BASE MODEL IS THE DATA VALIDATION ENGINE
    """
    PICO-lite decomposition of a medical query.
    
    Attributes:
        population: Disease/condition terms (P in PICO).
        intervention: Treatment/intervention/topic terms (I in PICO).
        comparison: Comparator terms (C in PICO) - optional.
        outcome: Outcome terms (O in PICO) - optional.
        modifiers: Additional qualifiers (stage, age, etc.).
        study_types: Optional study type filters.
        date_range: Optional publication date range (start_year, end_year).
        humans_only: Whether to filter for human studies.
    """
    # the field helper function of the pydantic always reconfigure the list  for each query 
    population: List[str] = Field(  # pydantic helper function to configure the field 
        default_factory=list,
        description="Disease/condition/population terms"   #  population=123(wrong) must be some disease name
    )
    intervention: List[str] = Field(
        default_factory=list,
        description="Intervention/treatment/topic terms"   # the description contains additional isntruction to the llm 
    )
    comparison: List[str] = Field(
        default_factory=list,
        description="Comparator terms (optional)"
    )
    outcome: List[str] = Field(
        default_factory=list,
        description="Outcome terms (optional)"
    )
    modifiers: List[str] = Field(
        default_factory=list,
        description="Modifier terms (stage, severity, etc.)"
    )
    study_types: List[str] = Field(
        default_factory=list,
        description="Study type filters (Clinical Trial, RCT, etc.)"
    )
    date_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Publication date range (start_year, end_year)"
    )
    humans_only: bool = Field(
        default=True,
        description="Filter for human studies"
    )

#################################################################################################################################
# =============================================================================
# Normalization Dictionaries (Controlled Vocabulary)
# =============================================================================
'''
====>These exist to translate intent → PubMed

=====>They prevent missing key papers

======>They embed domain knowledge

=======>They make RAG reliable

=======>Without them, answers degrade badly'''
# Common topic/intervention normalizations
#just like if user say treatment it will be normalized to therapy, management, Therapeutics[MeSH]
# and MESH IS MEDICAL SUBJECT HEADING  ( THE MAIN INDEXING CRITERIA OF THE PUBMED )

#“Search for the word melanoma only in the Title and Abstract of papers. OR WHERE THE MATCH HAPPENS THIS INCREASES THE RECALL ”
TOPIC_NORMALIZATIONS: Dict[str, List[str]] = {
    "treatment": ["treatment", "therapy", "management", "Therapeutics[MeSH]"],
    "diagnosis": ["diagnosis", "diagnostic", "screening", "detection", "Diagnosis[MeSH]"],
    "prognosis": ["prognosis", "survival", "outcome", "mortality", "Prognosis[MeSH]"],
    "prevention": ["prevention", "prophylaxis", "preventive", "Prevention and Control[MeSH]"],
    "etiology": ["etiology", "cause", "pathogenesis", "risk factors", "Etiology[MeSH]"],
    "epidemiology": ["epidemiology", "prevalence", "incidence", "Epidemiology[MeSH]"],
    "side effects": ["side effects", "adverse effects", "toxicity", "complications"],
    "guidelines": ["guidelines", "recommendations", "consensus", "Practice Guidelines[pt]"],
}

# Common disease MeSH mappings (expandable)
# COMMONG DISEASE NAMES NORMALIZATIONS 
DISEASE_MESH_MAP: Dict[str, str] = {
    "melanoma": "Melanoma[MeSH]",
    "diabetes": "Diabetes Mellitus[MeSH]",
    "hypertension": "Hypertension[MeSH]",
    "cancer": "Neoplasms[MeSH]",
    "heart failure": "Heart Failure[MeSH]",
    "asthma": "Asthma[MeSH]",
    "copd": "Pulmonary Disease, Chronic Obstructive[MeSH]",
    "alzheimer": "Alzheimer Disease[MeSH]",
    "parkinson": "Parkinson Disease[MeSH]",
    "stroke": "Stroke[MeSH]",
    "covid": "COVID-19[MeSH]",
    "covid-19": "COVID-19[MeSH]",
    "depression": "Depressive Disorder[MeSH]",
    "anxiety": "Anxiety Disorders[MeSH]",
    "arthritis": "Arthritis[MeSH]",
    "osteoporosis": "Osteoporosis[MeSH]",
    "pneumonia": "Pneumonia[MeSH]",
    "sepsis": "Sepsis[MeSH]",
    "leukemia": "Leukemia[MeSH]",
    "lymphoma": "Lymphoma[MeSH]",
}

#  COMMON STUDY TYPE NORMALIZATIONS AND PUBLICATION TYPE NORMALIZATIONS  #
# Study type publication type tags
STUDY_TYPE_TAGS: Dict[str, str] = {
    "clinical trial": "Clinical Trial[pt]",
    "rct": "Randomized Controlled Trial[pt]",
    "randomized controlled trial": "Randomized Controlled Trial[pt]",
    "meta-analysis": "Meta-Analysis[pt]",
    "systematic review": "Systematic Review[pt]",
    "case report": "Case Reports[pt]",
    "review": "Review[pt]",
    "guideline": "Practice Guideline[pt]",
}


# =============================================================================
# Query Builder
# =============================================================================

class PubMedQueryBuilder:
    """
    Builds high-recall, high-precision PubMed queries from PICO components.
    
    Uses:
    - MeSH terms with free-text fallback
    - Proper Boolean assembly (OR within concepts, AND between)
    - Title/Abstract field tags for precision
    - Optional precision boosters (study type, date, species)
    """
    
    def __init__(
        self,
        use_mesh: bool = True,  # Whether to include MeSH terms. AND SOME TIME WE NEED FREE TEXT OR DEBUG BEHAVIOUR USING THINGS CAN HELP OUT 
        use_tiab: bool = True,  # Whether to include title/abstract field tags.
        default_humans_filter: bool = True,  #ADDED TO HUMAN SPECIFIC FILTERS 
    ):
        """
        Initialize query builder.
        
        Args:
            use_mesh: Include MeSH terms in queries.
            use_tiab: Include title/abstract field tags.
            default_humans_filter: Add Humans[MeSH] filter by default.
        """
        self.use_mesh = use_mesh  #REFFERENCE THE CURRENT OBJECT 
        self.use_tiab = use_tiab
        self.default_humans_filter = default_humans_filter
        #EASY LOGGING AND DEBUGGING 
        logger.info(
            "PubMedQueryBuilder initialized",
            extra={"use_mesh": use_mesh, "use_tiab": use_tiab}
        )
    # BUILDS UP THE PUBMED QUERY FROM THE PICO COMPONENTS
    #PUBLIC FUNCTION THAT ACTUALLY CALLED 
    # THIS IS THE MAIN FUCNTION OF HOW THE QUERY IS BUILT UP 
    def build_query(self, pico: PICOQuery) -> str:
        """
        Build a PubMed query from PICO components.
        
        Args:
            pico: PICO query decomposition.
        
        Returns:
            Formatted PubMed query string.
        
        Example:
            >>> pico = PICOQuery(
            ...     population=["melanoma"],
            ...     intervention=["treatment"],
            ...     modifiers=["stage II"]
            ... )
            >>> builder.build_query(pico)
            '(("Melanoma"[MeSH] OR melanoma[tiab])) AND ...'
        """
        query_parts = []  # A EMPTY LIST INITIALIZATIONS 
        
        # Population/Disease block
        if pico.population:
            population_block = self._build_concept_block(
                terms=pico.population,
                mesh_map=DISEASE_MESH_MAP,
                concept_name="population"
            )
            query_parts.append(population_block)
        
        # Intervention/Topic block
        if pico.intervention:
            intervention_block = self._build_concept_block(
                terms=pico.intervention,
                normalization_map=TOPIC_NORMALIZATIONS,
                concept_name="intervention"
            )
            query_parts.append(intervention_block)
        
        # Comparison block (optional)
        if pico.comparison:
            comparison_block = self._build_concept_block(
                terms=pico.comparison,
                concept_name="comparison"
            )
            query_parts.append(comparison_block)
        
        # Outcome block (optional)
        if pico.outcome:
            outcome_block = self._build_concept_block(
                terms=pico.outcome,
                concept_name="outcome"
            )
            query_parts.append(outcome_block)
        
        # Modifier block
        #Additional qualifiers (stage, age, severity).
        #Why used:
        #Medical meaning depends heavily on modifiers.
        if pico.modifiers:
            modifier_block = self._build_modifier_block(pico.modifiers)
            query_parts.append(modifier_block)
        
        # Combine with AND
        main_query = " AND ".join(query_parts)  # JOING THE MAIN QUERY PARTS
        
        # Add precision boosters
        boosters = []
        
        # Study type filters
        if pico.study_types:
            study_filter = self._build_study_type_filter(pico.study_types)
            if study_filter:
                boosters.append(study_filter)
        
        # Date filter
        if pico.date_range:
            date_filter = self._build_date_filter(pico.date_range)
            boosters.append(date_filter)
        
        # Humans filter
        if pico.humans_only or self.default_humans_filter:
            boosters.append("Humans[MeSH]")
        
        # Combine main query with boosters
        if boosters:
            main_query = f"({main_query}) AND ({' AND '.join(boosters)})"
        
        logger.info(
            "Query built",
            extra={
                "population_count": len(pico.population),
                "intervention_count": len(pico.intervention),
                "query_length": len(main_query)
            }
        )
        
        return main_query

     ###########################################HELPER FUCNTIONS #######################################################################   
    # THIS ARE THE HELPER FUNCTION THAT ARE CALLED BY THE BUILD_QUERY FUNCTION
    def _build_concept_block(
        self,
        terms: List[str],
        mesh_map: Optional[Dict[str, str]] = None,
        normalization_map: Optional[Dict[str, List[str]]] = None,
        concept_name: str = "concept",
    ) -> str:
        """
        Build a concept block with OR-joined terms.
        
        Applies MeSH injection and term normalization.
        """
        expanded_terms = []
        
        for term in terms:
            term_lower = term.lower().strip()
            
            # Check for MeSH mapping
            if self.use_mesh and mesh_map and term_lower in mesh_map:
                mesh_term = mesh_map[term_lower]
                expanded_terms.append(mesh_term)
            
            # Check for normalization expansion
            if normalization_map and term_lower in normalization_map:
                for normalized in normalization_map[term_lower]:
                    if "[MeSH]" in normalized or "[pt]" in normalized:
                        expanded_terms.append(normalized)
                    else:
                        expanded_terms.append(self._format_tiab(normalized))
            else:
                # Add as title/abstract term
                expanded_terms.append(self._format_tiab(term))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for t in expanded_terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_terms.append(t)
        
        # Join with OR
        block = " OR ".join(unique_terms)
        
        return f"({block})"
    
    def _build_modifier_block(self, modifiers: List[str]) -> str:
        """
        Build a modifier block with variant handling.
        
        Handles common variations like "stage II" vs "stage 2".
        """
        expanded_modifiers = []
        
        for mod in modifiers:
            mod_clean = mod.strip()
            
            # Add original
            expanded_modifiers.append(self._format_tiab(mod_clean))
            
            # Handle Roman numeral variants
            variants = self._get_numeral_variants(mod_clean)
            for variant in variants:
                expanded_modifiers.append(self._format_tiab(variant))
        
        # Remove duplicates
        unique_mods = list(dict.fromkeys(expanded_modifiers))
        
        block = " OR ".join(unique_mods)
        return f"({block})"
    
    def _format_tiab(self, term: str) -> str:
        """Format a term with title/abstract field tag."""
        # Quote multi-word terms
        if " " in term and not term.startswith('"'):
            term = f'"{term}"'
        
        if self.use_tiab:
            return f"{term}[tiab]"
        return term
    
    def _get_numeral_variants(self, text: str) -> List[str]:
        """
        Generate Roman/Arabic numeral variants.
        
        Examples:
            "stage II" -> ["stage 2"]
            "stage 2" -> ["stage II"]
            "type 1" -> ["type I"]
        """
        variants = []
        
        # Roman to Arabic mapping
        roman_to_arabic = {
            "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
            "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10"
        }
        arabic_to_roman = {v: k for k, v in roman_to_arabic.items()}
        
        # Check for Roman numerals (case insensitive)
        for roman, arabic in roman_to_arabic.items():
            pattern = rf'\b{roman}\b'
            if re.search(pattern, text, re.IGNORECASE):
                variant = re.sub(pattern, arabic, text, flags=re.IGNORECASE)
                if variant != text:
                    variants.append(variant)
        
        # Check for Arabic numerals
        for arabic, roman in arabic_to_roman.items():
            pattern = rf'\b{arabic}\b'
            if re.search(pattern, text):
                variant = re.sub(pattern, roman, text)
                if variant != text:
                    variants.append(variant)
        
        return variants
    
    def _build_study_type_filter(self, study_types: List[str]) -> str:
        """Build study type publication type filter."""
        type_tags = []
        
        for st in study_types:
            st_lower = st.lower().strip()
            if st_lower in STUDY_TYPE_TAGS:
                type_tags.append(STUDY_TYPE_TAGS[st_lower])
            else:
                # Try as-is with [pt] tag
                type_tags.append(f"{st}[pt]")
        
        if type_tags:
            return f"({' OR '.join(type_tags)})"
        return ""
    
    def _build_date_filter(self, date_range: Tuple[int, int]) -> str:
        """Build publication date filter."""
        start_year, end_year = date_range
        return f'("{start_year}"[dp] : "{end_year}"[dp])'
    
    def normalize_topic(self, topic: str) -> List[str]:
        """
        Expand a topic term using normalization dictionary.
        
        Args:
            topic: Raw topic term.
        
        Returns:
            List of normalized/expanded terms.
        """
        topic_lower = topic.lower().strip()
        
        if topic_lower in TOPIC_NORMALIZATIONS:
            return TOPIC_NORMALIZATIONS[topic_lower]
        
        return [topic]
    
    def get_mesh_term(self, disease: str) -> Optional[str]:
        """
        Get MeSH term for a disease if available.
        
        Args:
            disease: Disease name.
        
        Returns:
            MeSH term string or None.
        """
        disease_lower = disease.lower().strip()
        return DISEASE_MESH_MAP.get(disease_lower)

''' 
# =============================================================================
# LLM-based PICO Decomposition Schema
# =============================================================================

class PICODecomposition(BaseModel):
    """
    Schema for LLM-based PICO decomposition of free-text queries.
    
    Used with Pydantic AI to structure LLM output.
    """
    population_terms: List[str] = Field(
        description="Disease, condition, or patient population terms extracted from the query"
    )
    intervention_terms: List[str] = Field(
        description="Treatment, intervention, diagnostic, or topic terms"
    )
    comparison_terms: List[str] = Field(
        default_factory=list,
        description="Comparator terms if query involves comparison"
    )
    outcome_terms: List[str] = Field(
        default_factory=list,
        description="Outcome or endpoint terms"
    )
    modifier_terms: List[str] = Field(
        default_factory=list,
        description="Modifiers like stage, severity, age group, etc."
    )
    suggested_study_types: List[str] = Field(
        default_factory=list,
        description="Suggested study types based on query intent (e.g., 'RCT', 'systematic review')"
    )
    requires_recent: bool = Field(
        default=False,
        description="Whether the query implies need for recent/latest publications"
    )
    
    def to_pico_query(self, recent_years: int = 5) -> PICOQuery:
        """Convert decomposition to PICOQuery."""
        import datetime
        
        date_range = None
        if self.requires_recent:
            current_year = datetime.datetime.now().year
            date_range = (current_year - recent_years, current_year)
        
        return PICOQuery(
            population=self.population_terms,
            intervention=self.intervention_terms,
            comparison=self.comparison_terms,
            outcome=self.outcome_terms,
            modifiers=self.modifier_terms,
            study_types=self.suggested_study_types,
            date_range=date_range,
            humans_only=True,
        )


# =============================================================================
# Quick Helper Functions
# =============================================================================

def build_simple_query(
    disease: str,
    topic: str,
    modifiers: Optional[List[str]] = None,
    years: Optional[int] = None,
) -> str:
    """
    Quick helper to build a simple PubMed query.
    
    Args:
        disease: Disease/condition term.
        topic: Topic/intervention term.
        modifiers: Optional modifier terms.
        years: Optional limit to recent N years.
    
    Returns:
        PubMed query string.
    
    Example:
        #>>> build_simple_query("melanoma", "treatment", ["stage II"], years=5)
        '(("Melanoma"[MeSH] OR melanoma[tiab])) AND ...'
    """
    import datetime
    
    date_range = None
    if years:
        current_year = datetime.datetime.now().year
        date_range = (current_year - years, current_year)
    
    pico = PICOQuery(
        population=[disease],
        intervention=[topic],
        modifiers=modifiers or [],
        date_range=date_range,
    )
    
    builder = PubMedQueryBuilder()
    return builder.build_query(pico)
'''