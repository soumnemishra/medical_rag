# FILE: src/pubmed_client.py
# this file basically is resposible for fetches the data from the pub med and then parse them clean them 
# and returns the validated #document that we can feed to our llms 

# it is the data ingestion and validation layer of the base rag system 
# this goes in this sequence 
# e seach --> e fetch --> filtering --> validation --> llm 
''' if we take into consideration of the medical hosptital 
e search --> search the patents
e fetch --> get the full documents

filtering --> remove irrlelvant cases
validate --> check reports are usable 
llm ready documents 
'''

"""
PubMed E-utilities client with 2-stage retrieval pipeline.

Implements a production-grade retrieval system:
    Doctor Query → Structured Query → eSearch → eFetch → Filtering → Pydantic Objects

Architecture:
    1. eSearch: Query → PMIDs (JSON response)
    2. eFetch: PMIDs → Full articles (XML response)
    3. Post-retrieval filtering (Humans, date, study type) # it filers out the human studies 
    4. Pydantic validation layer

Example Usage:
    from src.pubmed_client import PubMedClient
    
    client = PubMedClient()
    results = client.search("diabetes treatment")
    for doc in results:
        print(doc.pmid, doc.title)
"""

import logging # we log important events for easy debugging and monitoring 
from dataclasses import dataclass, field
from datetime import datetime # for date time filtering of articles 
from enum import Enum # enum for study types 
from typing import List, Optional, Set

import requests #http client for making request to pubmed api
from bs4 import BeautifulSoup #used to traverse the xml data returned by the pubmed 
# this ensure strict check this ensures that data entring to agents is strictly validated and clean 
#if pub med return the garbage data or empty data the pydantic will catches the garbage before it reaches the llm
from pydantic import BaseModel, Field, field_validator # data validations The star of the validation layer. 
#It ensures every document we produce is clean, with correct types and non-empty abstracts.

# if api fails then try again intelligetly 
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
# import the configuration s setting 
from src.config import settings
from src.exceptions import PubMedAPIError, TransientError

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class StudyType(str, Enum):

    """Accepted study types for clinical relevance.
    this tells that we accept this types of studies only for our rag system and we will filter out the rest of the studies
    the enum helps us to have a predefined set of study types that we consider relevant for our use case.
    and removes the typo"""
    CLINICAL_TRIAL = "Clinical Trial"
    RCT = "Randomized Controlled Trial"
    SYSTEMATIC_REVIEW = "Systematic Review"
    META_ANALYSIS = "Meta-Analysis"
    REVIEW = "Review"
    CASE_REPORT = "Case Reports"
    GUIDELINE = "Practice Guideline"
    OBSERVATIONAL = "Observational Study"


ALLOWED_STUDY_TYPES: Set[str] = {st.value for st in StudyType}


# =============================================================================
# Pydantic Models (Validation Layer)
# =============================================================================

# this is the data validation layer of how should the retreive document should look like and what are the require fields
# this is being validated by the pydantic base model and the field validators 
class RetrievedDocument(BaseModel):
    """
    Validated PubMed document ready for RAG pipeline.
    
    This is the FINAL output - clean, validated, and safe.
    
    Attributes:
        pmid: PubMed identifier (unique).
        title: Article title.
        abstract: Full abstract text.
        year: Publication year.
        authors: List of author names.
        journal: Journal name.
        doi: Digital Object Identifier (optional).
        study_types: Publication types.
        mesh_terms: MeSH descriptor terms.
        source: Always "PubMed".
    """
    pmid: str = Field(description="PubMed ID")
    title: str = Field(description="Article title")
    abstract: str = Field(min_length=1, description="Abstract text")
    year: Optional[int] = Field(default=None, description="Publication year")
    authors: List[str] = Field(default_factory=list, description="Author names")
    journal: str = Field(default="", description="Journal name")
    doi: Optional[str] = Field(default=None, description="DOI")
    study_types: List[str] = Field(default_factory=list, description="Publication types")
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH terms")
    source: str = Field(default="PubMed", description="Data source")
    
    # the bouncer the pub med sometimes return the emppty abstract or some papers have 
    #whose xm have the title, authors , pmid , but the <abstract> tag is completely empty 
    @field_validator("abstract")
    @classmethod
    def abstract_not_empty(cls, v: str) -> str:
        """Reject articles without abstracts.
        it says if no abstracts or not v.strip() then raise the value error """
        if not v or not v.strip():
            raise ValueError("Abstract cannot be empty")
        return v.strip()
    
    ### it 
    def to_context_string(self) -> str:
        """Format document as context string for LLM."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        
        study_str = ", ".join(self.study_types[:2]) if self.study_types else "Article"
        
        return (
            f"**{self.title}**\n"
            f"Authors: {authors_str}\n"
            f"Journal: {self.journal} ({self.year or 'N/A'})\n"
            f"Type: {study_str}\n"
            f"PMID: {self.pmid}\n\n"
            f"{self.abstract}\n"
        )
    def to_citation(self) -> str:
        """Format as citation."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{authors_str}. {self.title}. {self.journal}. {self.year or 'N/A'}. PMID: {self.pmid}"


class SearchMetadata(BaseModel):
    """Metadata from PubMed search."""
    total_count: int = Field(description="Total results in PubMed")
    returned_count: int = Field(description="PMIDs returned")
    query_translation: str = Field(default="", description="PubMed's query translation")


# =============================================================================
# Raw Article (Pre-validation)

##################why we use th data classs lets discuss ##########################
''' we use the data class beacuse xml parsing is messy , when u pull data out of beautiful soup it might be 
incomplete or misformatted if we tried to solve that raw xml data with pydantic it would be a nightmare because 
pydantic is strict and expects clean data
'''
# =============================================================================

@dataclass
class RawArticle:
    """
    Raw parsed article before validation.
    
    This is an intermediate format between XML parsing and Pydantic validation.
    """
    pmid: str
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    doi: Optional[str] = None
    study_types: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    is_human_study: bool = False
#That single = False protects your entire RAG pipeline from being poisoned by 
# irrelevant biological studies. It ensures your baseline evaluation over PubMedQA 
# will be grounded only in verified human clinical literature.

# =============================================================================
# PubMed Client
# =============================================================================

class PubMedClient:
    """
    Production-grade PubMed client with 2-stage retrieval.
    
    Pipeline:
        Query → eSearch (PMIDs) → eFetch (XML) → Parse → Filter → Validate
    
    Features:
        - Separate search and fetch stages
        - Post-retrieval filtering (humans, date, study type)
        - Pydantic validation layer
        - Exponential backoff retry
        - API key support for higher rate limits
    """
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: Optional[int] = None,
        filter_humans: bool = True,
        filter_recent_years: Optional[int] = None,
        filter_study_types: bool = False,
    ):
        """
        Initialize PubMed client.
        
        Args:
            api_key: NCBI API key for higher rate limits.
            max_results: Maximum PMIDs to retrieve per search.
            filter_humans: Filter for human studies only.
            filter_recent_years: Filter to last N years (None to disable).
            filter_study_types: Filter for clinical study types only.
        """
        self.api_key = api_key or settings.PUBMED_API_KEY
        self.max_results = max_results or settings.MAX_SEARCH_RESULTS
        self.filter_humans = filter_humans
        self.filter_recent_years = filter_recent_years
        self.filter_study_types = filter_study_types
        self._session = requests.Session()
        
        logger.info(
            "PubMed client initialized",
            extra={
                "max_results": self.max_results,
                "filter_humans": filter_humans,
                "filter_recent_years": filter_recent_years,
            }
        )
    
    def _get_base_params(self) -> dict:
        """Get base parameters including API key."""
        params = {"db": "pubmed"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    # =========================================================================
    # STAGE 1: eSearch - Query → PMIDs
    # =========================================================================
    
    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=settings.RETRY_WAIT_SECONDS, max=10),
        retry=retry_if_exception_type(TransientError),
        reraise=True,
    )
    def esearch(self, query: str, max_results: Optional[int] = None) -> tuple[List[str], SearchMetadata]:
        """
        Stage 1: Search PubMed and get PMIDs.
        
        Args:
            query: PubMed query string.
            max_results: Override default max results.
        
        Returns:
            Tuple of (list of PMIDs, search metadata).
        
        Raises:
            PubMedAPIError: On API failure.
        """
        params = self._get_base_params()
        params.update({
            "term": query,
            "retmax": max_results or self.max_results,
            "retmode": "json",
            "sort": "relevance",
        })
        
        logger.info("eSearch", extra={"query": query[:100]})
        
        try:
            response = self._session.get(
                self.ESEARCH_URL,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
        except requests.exceptions.Timeout as e:
            raise PubMedAPIError("eSearch timeout", {"query": query}) from e
        except requests.exceptions.RequestException as e:
            raise PubMedAPIError(f"eSearch failed: {e}", {"query": query}) from e
        
        try:
            data = response.json()
            result = data.get("esearchresult", {})
            
            pmids = result.get("idlist", [])
            metadata = SearchMetadata(
                total_count=int(result.get("count", 0)),
                returned_count=len(pmids),
                query_translation=result.get("querytranslation", ""),
            )
            
            logger.info(
                "eSearch complete",
                extra={
                    "pmid_count": len(pmids),
                    "total_available": metadata.total_count,
                }
            )
            
            return pmids, metadata
            
        except (KeyError, ValueError) as e:
            raise PubMedAPIError("Failed to parse eSearch response", {"error": str(e)}) from e
    
    # =========================================================================
    # STAGE 2: eFetch - PMIDs → Articles
    # =========================================================================
    
    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=settings.RETRY_WAIT_SECONDS, max=10),
        retry=retry_if_exception_type(TransientError),
        reraise=True,
    )
    def efetch(self, pmids: List[str]) -> List[RawArticle]:
        """
        Stage 2: Fetch full article data for PMIDs.
        
        Args:
            pmids: List of PubMed IDs.
        
        Returns:
            List of RawArticle objects (pre-validation).
        
        Raises:
            PubMedAPIError: On API failure.
        """
        if not pmids:
            return []
        
        params = self._get_base_params()
        params.update({
            "id": ",".join(pmids),
            "retmode": "xml",
        })
        
        logger.info("eFetch", extra={"pmid_count": len(pmids)})
        
        try:
            response = self._session.get(
                self.EFETCH_URL,
                params=params,
                timeout=60
            )
            response.raise_for_status()
            
        except requests.exceptions.Timeout as e:
            raise PubMedAPIError("eFetch timeout", {"pmids": pmids[:5]}) from e
        except requests.exceptions.RequestException as e:
            raise PubMedAPIError(f"eFetch failed: {e}", {"pmids": pmids[:5]}) from e
        
        # Parse XML
        articles = self._parse_xml(response.content)
        
        logger.info("eFetch complete", extra={"articles_parsed": len(articles)})
        
        return articles
    
    # =========================================================================
    # XML Parsing (Using BeautifulSoup)
    # =========================================================================
    
    def _parse_xml(self, xml_content: bytes) -> List[RawArticle]:
        """Parse PubMed XML response into RawArticle objects."""
        articles = []
        
        try:
            soup = BeautifulSoup(xml_content, "xml")
            
            for article_elem in soup.find_all("PubmedArticle"):
                try:
                    article = self._parse_single_article(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(
                        "Failed to parse article",
                        extra={"error": str(e)}
                    )
                    continue
                    
        except Exception as e:
            logger.error("XML parse error", extra={"error": str(e)})
            
        return articles
    
    def _parse_single_article(self, article_elem) -> Optional[RawArticle]:
        """Parse a single PubmedArticle element."""
        # PMID (required)
        pmid_elem = article_elem.find("PMID")
        if not pmid_elem or not pmid_elem.text:
            return None
        pmid = pmid_elem.text.strip()
        
        # Title
        title_elem = article_elem.find("ArticleTitle")
        title = title_elem.text.strip() if title_elem and title_elem.text else ""
        
        # Abstract (combine all AbstractText elements)
        abstract_parts = []
        for abs_elem in article_elem.find_all("AbstractText"):
            if abs_elem.text:
                label = abs_elem.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {abs_elem.text.strip()}")
                else:
                    abstract_parts.append(abs_elem.text.strip())
        abstract = " ".join(abstract_parts)
        
        # Year
        year = None
        pub_date = article_elem.find("PubDate")
        if pub_date:
            year_elem = pub_date.find("Year")
            if year_elem and year_elem.text:
                try:
                    year = int(year_elem.text.strip())
                except ValueError:
                    pass
        
        # Authors
        authors = []
        for author in article_elem.find_all("Author"):
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname and lastname.text:
                name = lastname.text.strip()
                if forename and forename.text:
                    name = f"{forename.text.strip()} {name}"
                authors.append(name)
        
        # Journal
        journal = ""
        journal_elem = article_elem.find("Journal")
        if journal_elem:
            title_elem = journal_elem.find("Title")
            if title_elem and title_elem.text:
                journal = title_elem.text.strip()
        
        # DOI
        doi = None
        for id_elem in article_elem.find_all("ArticleId"):
            if id_elem.get("IdType") == "doi" and id_elem.text:
                doi = id_elem.text.strip()
                break
        
        # Study Types (Publication Types)
        study_types = []
        for pub_type in article_elem.find_all("PublicationType"):
            if pub_type.text:
                study_types.append(pub_type.text.strip())
        
        # MeSH Terms
        mesh_terms = []
        for mesh in article_elem.find_all("DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text.strip())
        
        # Check if human study
        is_human = "Humans" in mesh_terms
        
        return RawArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            journal=journal,
            doi=doi,
            study_types=study_types,
            mesh_terms=mesh_terms,
            is_human_study=is_human,
        )
    
    # =========================================================================
    # STAGE 3: Post-Retrieval Filtering
    # =========================================================================
    
    def filter_articles(self, articles: List[RawArticle]) -> List[RawArticle]:
        """
        Apply post-retrieval filters.
        
        Filters applied:
            1. Humans only (if enabled)
            2. Recent years (if configured)
            3. Clinical study types (if enabled)
        
        Args:
            articles: Raw articles to filter.
        
        Returns:
            Filtered list of articles.
        """
        filtered = articles
        initial_count = len(articles)
        
        # Filter: Humans only
        if self.filter_humans:
            filtered = [a for a in filtered if a.is_human_study]
            logger.debug(f"After humans filter: {len(filtered)}/{initial_count}")
        
        # Filter: Recent years
        if self.filter_recent_years:
            current_year = datetime.now().year
            cutoff = current_year - self.filter_recent_years
            filtered = [
                a for a in filtered
                if a.year is None or a.year >= cutoff
            ]
            logger.debug(f"After date filter: {len(filtered)}/{initial_count}")
        
        # Filter: Study types
        if self.filter_study_types:
            filtered = [
                a for a in filtered
                if any(st in ALLOWED_STUDY_TYPES for st in a.study_types)
            ]
            logger.debug(f"After study type filter: {len(filtered)}/{initial_count}")
        
        logger.info(
            "Filtering complete",
            extra={"before": initial_count, "after": len(filtered)}
        )
        
        return filtered
    
    # =========================================================================
    # STAGE 4: Pydantic Validation
    # =========================================================================
    
    def validate_articles(self, articles: List[RawArticle]) -> List[RetrievedDocument]:
        """
        Validate articles through Pydantic layer.
        
        Articles without abstracts or with invalid data are rejected.
        
        Args:
            articles: Raw articles to validate.
        
        Returns:
            List of validated RetrievedDocument objects.
        """
        validated = []
        rejected = 0
        
        for article in articles:
            try:
                doc = RetrievedDocument(
                    pmid=article.pmid,
                    title=article.title,
                    abstract=article.abstract,
                    year=article.year,
                    authors=article.authors,
                    journal=article.journal,
                    doi=article.doi,
                    study_types=article.study_types,
                    mesh_terms=article.mesh_terms,
                )
                validated.append(doc)
            except Exception as e:
                rejected += 1
                logger.debug(
                    "Article rejected by validation",
                    extra={"pmid": article.pmid, "reason": str(e)}
                )
        
        logger.info(
            "Validation complete",
            extra={"validated": len(validated), "rejected": rejected}
        )
        
        return validated
    
    # =========================================================================
    # Main Search Method (Full Pipeline)
    # =========================================================================
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[RetrievedDocument]:
        """
        Execute full 2-stage retrieval pipeline.
        
        Pipeline:
            Query → eSearch → eFetch → Filter → Validate → RetrievedDocuments
        
        Args:
            query: PubMed query string.
            max_results: Override default max results.
        
        Returns:
            List of validated RetrievedDocument objects.
        
        Example:
            >>> client = PubMedClient()
            >>> docs = client.search("melanoma treatment")
            >>> for doc in docs:
            ...     print(f"PMID:{doc.pmid} - {doc.title}")
        """
        logger.info("Starting search pipeline", extra={"query": query[:100]})
        
        # Stage 1: Search for PMIDs
        pmids, metadata = self.esearch(query, max_results)
        
        if not pmids:
            logger.info("No PMIDs found", extra={"query": query[:50]})
            return []
        
        # Stage 2: Fetch articles
        raw_articles = self.efetch(pmids)
        
        if not raw_articles:
            logger.warning("No articles fetched", extra={"pmid_count": len(pmids)})
            return []
        
        # Stage 3: Filter
        filtered_articles = self.filter_articles(raw_articles)
        
        # Stage 4: Validate
        validated_docs = self.validate_articles(filtered_articles)
        
        logger.info(
            "Search pipeline complete",
            extra={
                "query": query[:50],
                "pmids": len(pmids),
                "parsed": len(raw_articles),
                "filtered": len(filtered_articles),
                "validated": len(validated_docs),
            }
        )
        
        return validated_docs
    
    def search_with_metadata(
        self, query: str, max_results: Optional[int] = None
    ) -> tuple[List[RetrievedDocument], SearchMetadata]:
        """
        Search with metadata about the search.
        
        Returns:
            Tuple of (documents, search metadata).
        """
        pmids, metadata = self.esearch(query, max_results)
        
        if not pmids:
            return [], metadata
        
        raw_articles = self.efetch(pmids)
        filtered_articles = self.filter_articles(raw_articles)
        validated_docs = self.validate_articles(filtered_articles)
        
        return validated_docs, metadata


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Alias for backward compatibility with existing code
PubMedArticle = RetrievedDocument
