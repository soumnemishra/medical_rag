# FILE: tests/test_pubmed_client.py
"""
Unit tests for 2-stage PubMed retrieval pipeline.

Tests eSearch, eFetch, filtering, and Pydantic validation.
Uses mocked HTTP responses - no real network calls.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from src.pubmed_client import (
    PubMedClient,
    RetrievedDocument,
    RawArticle,
    SearchMetadata,
    ALLOWED_STUDY_TYPES,
)
from src.exceptions import PubMedAPIError


# =============================================================================
# Sample API Responses for Mocking
# =============================================================================

SAMPLE_ESEARCH_RESPONSE = {
    "esearchresult": {
        "count": "150",
        "idlist": ["12345678", "87654321", "11111111"],
        "querytranslation": "melanoma[MeSH] AND treatment[tiab]",
    }
}

SAMPLE_EFETCH_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Treatment advances in stage II melanoma</ArticleTitle>
                <Abstract>
                    <AbstractText Label="BACKGROUND">Melanoma is a serious condition.</AbstractText>
                    <AbstractText Label="METHODS">We conducted a randomized trial.</AbstractText>
                    <AbstractText Label="RESULTS">Treatment was effective.</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                    <Author>
                        <LastName>Johnson</LastName>
                        <ForeName>Jane</ForeName>
                    </Author>
                </AuthorList>
                <Journal>
                    <Title>Journal of Clinical Oncology</Title>
                </Journal>
                <PublicationTypeList>
                    <PublicationType>Randomized Controlled Trial</PublicationType>
                    <PublicationType>Clinical Trial</PublicationType>
                </PublicationTypeList>
            </Article>
        </MedlineCitation>
        <PubmedData>
            <History>
                <PubMedPubDate PubStatus="pubmed">
                    <Year>2024</Year>
                    <Month>06</Month>
                </PubMedPubDate>
            </History>
            <ArticleIdList>
                <ArticleId IdType="doi">10.1234/jco.2024.001</ArticleId>
            </ArticleIdList>
        </PubmedData>
        <MeshHeadingList>
            <MeshHeading>
                <DescriptorName>Melanoma</DescriptorName>
            </MeshHeading>
            <MeshHeading>
                <DescriptorName>Humans</DescriptorName>
            </MeshHeading>
        </MeshHeadingList>
    </PubmedArticle>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>87654321</PMID>
            <Article>
                <ArticleTitle>Mouse model of melanoma</ArticleTitle>
                <Abstract>
                    <AbstractText>Study in mouse models showed promising results.</AbstractText>
                </Abstract>
                <Journal>
                    <Title>Lab Animal Research</Title>
                </Journal>
            </Article>
        </MedlineCitation>
        <PubmedData>
            <History>
                <PubMedPubDate PubStatus="pubmed">
                    <Year>2023</Year>
                </PubMedPubDate>
            </History>
        </PubmedData>
        <MeshHeadingList>
            <MeshHeading>
                <DescriptorName>Melanoma</DescriptorName>
            </MeshHeading>
            <MeshHeading>
                <DescriptorName>Mice</DescriptorName>
            </MeshHeading>
        </MeshHeadingList>
    </PubmedArticle>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>11111111</PMID>
            <Article>
                <ArticleTitle>Article without abstract</ArticleTitle>
                <Journal>
                    <Title>Some Journal</Title>
                </Journal>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>
"""


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a PubMed client for testing."""
    return PubMedClient(
        filter_humans=True,
        filter_recent_years=7,
        filter_study_types=False,
    )


@pytest.fixture
def mock_esearch_response():
    """Create mock eSearch response."""
    response = AsyncMock()
    response.json.return_value = SAMPLE_ESEARCH_RESPONSE
    response.status = 200
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def mock_efetch_response():
    """Create mock eFetch response."""
    response = AsyncMock()
    response.read.return_value = SAMPLE_EFETCH_XML
    response.status = 200
    response.raise_for_status = Mock()
    return response


# Helper to mock aiohttp session
@pytest.fixture
def mock_aiohttp_session():
    with patch("src.pubmed_client.aiohttp.ClientSession") as mock_session_cls:
        # The context manager returned by ClientSession() constructor
        session_cm = AsyncMock()
        mock_session_cls.return_value = session_cm
        
        # The session object yielded by the context manager
        session = MagicMock()
        session_cm.__aenter__.return_value = session
        
        yield session


# =============================================================================
# Test: eSearch Stage
# =============================================================================

class TestESearch:
    """Tests for eSearch (Stage 1)."""
    
    @pytest.mark.asyncio
    async def test_esearch_returns_pmids(self, client, mock_aiohttp_session, mock_esearch_response):
        """Test that eSearch returns list of PMIDs."""
        # Setup mock return (context manager for get)
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_esearch_response
        mock_aiohttp_session.get.return_value = mock_get_ctx
        
        pmids, metadata = await client.esearch("melanoma treatment")
        
        assert len(pmids) == 3
        assert "12345678" in pmids
        assert metadata.total_count == 150
    
    @pytest.mark.asyncio
    async def test_esearch_empty_results(self, client, mock_aiohttp_session):
        """Test eSearch with no results."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "esearchresult": {"count": "0", "idlist": []}
        }
        mock_response.raise_for_status = Mock()
        
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_aiohttp_session.get.return_value = mock_get_ctx
        
        pmids, metadata = await client.esearch("nonexistent query xyz")
        
        assert pmids == []
        assert metadata.total_count == 0
    
    @pytest.mark.asyncio
    async def test_esearch_timeout_raises_error(self, client, mock_aiohttp_session):
        """Test that timeout raises PubMedAPIError."""
        import asyncio
        mock_aiohttp_session.get.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(PubMedAPIError) as exc_info:
            await client.esearch("test query")
        
        assert "timeout" in str(exc_info.value).lower()


# =============================================================================
# Test: eFetch Stage
# =============================================================================

class TestEFetch:
    """Tests for eFetch (Stage 2)."""
    
    @pytest.mark.asyncio
    async def test_efetch_parses_articles(self, client, mock_aiohttp_session, mock_efetch_response):
        """Test that eFetch parses XML correctly."""
        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_efetch_response
        mock_aiohttp_session.get.return_value = mock_get_ctx
        
        articles = await client.efetch(["12345678", "87654321"])
        
        assert len(articles) >= 2
        
        # Check first article
        article1 = next(a for a in articles if a.pmid == "12345678")
        assert "melanoma" in article1.title.lower()
        assert len(article1.authors) == 2
        assert article1.is_human_study is True
    
    @pytest.mark.asyncio
    async def test_efetch_empty_pmids(self, client, mock_aiohttp_session):
        """Test eFetch with empty PMID list."""
        articles = await client.efetch([])
        
        assert articles == []
        mock_aiohttp_session.get.assert_not_called()


# =============================================================================
# Test: Filtering Stage
# =============================================================================

class TestFiltering:
    """Tests for post-retrieval filtering (Stage 3)."""
    
    def test_filter_humans_only(self):
        """Test human study filter."""
        client = PubMedClient(filter_humans=True, filter_recent_years=None)
        
        articles = [
            RawArticle(pmid="1", title="Human study", abstract="test", is_human_study=True),
            RawArticle(pmid="2", title="Mouse study", abstract="test", is_human_study=False),
        ]
        
        filtered = client.filter_articles(articles)
        
        assert len(filtered) == 1
        assert filtered[0].pmid == "1"
    
    def test_filter_recent_years(self):
        """Test date filter."""
        client = PubMedClient(filter_humans=False, filter_recent_years=5)
        current_year = datetime.now().year
        
        articles = [
            RawArticle(pmid="1", abstract="test", year=current_year, is_human_study=True),
            RawArticle(pmid="2", abstract="test", year=current_year - 10, is_human_study=True),
            RawArticle(pmid="3", abstract="test", year=None, is_human_study=True),  # No year = keep
        ]
        
        filtered = client.filter_articles(articles)
        
        assert len(filtered) == 2
        assert any(a.pmid == "1" for a in filtered)
        assert any(a.pmid == "3" for a in filtered)
    
    def test_filter_study_types(self):
        """Test study type filter."""
        client = PubMedClient(
            filter_humans=False,
            filter_recent_years=None,
            filter_study_types=True,
        )
        
        articles = [
            RawArticle(pmid="1", abstract="test", study_types=["Randomized Controlled Trial"]),
            RawArticle(pmid="2", abstract="test", study_types=["Letter"]),
            RawArticle(pmid="3", abstract="test", study_types=["Review", "Meta-Analysis"]),
        ]
        
        filtered = client.filter_articles(articles)
        
        assert len(filtered) == 2
        assert any(a.pmid == "1" for a in filtered)
        assert any(a.pmid == "3" for a in filtered)


# =============================================================================
# Test: Pydantic Validation Stage
# =============================================================================

class TestValidation:
    """Tests for Pydantic validation (Stage 4)."""
    
    def test_validates_complete_article(self, client):
        """Test validation of complete article."""
        articles = [
            RawArticle(
                pmid="12345",
                title="Test Article",
                abstract="This is a valid abstract.",
                year=2024,
                authors=["John Smith"],
                journal="Test Journal",
            )
        ]
        
        validated = client.validate_articles(articles)
        
        assert len(validated) == 1
        assert isinstance(validated[0], RetrievedDocument)
        assert validated[0].pmid == "12345"
    
    def test_rejects_empty_abstract(self, client):
        """Test that articles without abstracts are rejected."""
        articles = [
            RawArticle(pmid="1", title="Test", abstract="Valid abstract"),
            RawArticle(pmid="2", title="Test", abstract=""),
            RawArticle(pmid="3", title="Test", abstract="   "),
        ]
        
        validated = client.validate_articles(articles)
        
        assert len(validated) == 1
        assert validated[0].pmid == "1"


# =============================================================================
# Test: Full Pipeline
# =============================================================================

class TestFullPipeline:
    """Tests for full search pipeline."""
    
    @pytest.mark.asyncio
    async def test_search_pipeline(
        self, client, mock_aiohttp_session, mock_esearch_response, mock_efetch_response
    ):
        """Test complete search pipeline."""
        # Setup mocks for sequential calls
        # 1. eSearch
        mock_esearch_ctx = AsyncMock()
        mock_esearch_ctx.__aenter__.return_value = mock_esearch_response
        
        # 2. eFetch
        mock_efetch_ctx = AsyncMock()
        mock_efetch_ctx.__aenter__.return_value = mock_efetch_response
        
        # Configure side_effect to return different contexts
        mock_aiohttp_session.get.side_effect = [mock_esearch_ctx, mock_efetch_ctx]
        
        results = await client.search("melanoma treatment")
        
        # Should have validated documents (only human study with abstract)
        assert len(results) >= 1
        assert all(isinstance(r, RetrievedDocument) for r in results)
        
        # Check that human study is included
        human_study = next((r for r in results if r.pmid == "12345678"), None)
        assert human_study is not None
        assert "Humans" in human_study.mesh_terms


# =============================================================================
# Test: RetrievedDocument Model
# =============================================================================

class TestRetrievedDocument:
    """Tests for RetrievedDocument Pydantic model."""
    
    def test_to_context_string(self):
        """Test context string generation."""
        doc = RetrievedDocument(
            pmid="12345",
            title="Test Article",
            abstract="This is the abstract.",
            year=2024,
            authors=["A One", "B Two", "C Three", "D Four"],
            journal="Medical Journal",
            study_types=["Clinical Trial"],
        )
        
        context = doc.to_context_string()
        
        assert "Test Article" in context
        assert "12345" in context
        assert "et al." in context  # >3 authors
        assert "Clinical Trial" in context
    
    def test_to_citation(self):
        """Test citation generation."""
        doc = RetrievedDocument(
            pmid="12345",
            title="Test Article",
            abstract="Abstract text.",
            year=2024,
            authors=["John Smith", "Jane Doe"],
            journal="Medical Journal",
        )
        
        citation = doc.to_citation()
        
        assert "John Smith" in citation
        assert "Test Article" in citation
        assert "PMID: 12345" in citation
