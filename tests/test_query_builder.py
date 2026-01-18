# FILE: tests/test_query_builder.py
"""
Unit tests for PICO-based PubMed query builder.
"""

import pytest
from src.query_builder import (
    PubMedQueryBuilder,
    PICOQuery,
    PICODecomposition,
    build_simple_query,
    DISEASE_MESH_MAP,
    TOPIC_NORMALIZATIONS,
)


class TestPICOQuery:
    """Tests for PICOQuery model."""
    
    def test_create_basic_pico(self):
        """Test creating a basic PICO query."""
        pico = PICOQuery(
            population=["melanoma"],
            intervention=["treatment"],
        )
        assert pico.population == ["melanoma"]
        assert pico.intervention == ["treatment"]
        assert pico.modifiers == []
    
    def test_create_full_pico(self):
        """Test creating a full PICO query with all fields."""
        pico = PICOQuery(
            population=["diabetes", "type 2 diabetes"],
            intervention=["metformin", "treatment"],
            comparison=["placebo"],
            outcome=["HbA1c reduction"],
            modifiers=["elderly", "obese"],
            study_types=["rct"],
            date_range=(2020, 2025),
            humans_only=True,
        )
        assert len(pico.population) == 2
        assert pico.date_range == (2020, 2025)


class TestPubMedQueryBuilder:
    """Tests for PubMedQueryBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create a query builder instance."""
        return PubMedQueryBuilder()
    
    def test_build_simple_query(self, builder):
        """Test building a simple query with population and intervention."""
        pico = PICOQuery(
            population=["melanoma"],
            intervention=["treatment"],
        )
        query = builder.build_query(pico)
        
        # Should contain MeSH term for melanoma
        assert "Melanoma[MeSH]" in query
        # Should contain treatment terms
        assert "treatment" in query.lower()
        # Should have AND between concepts
        assert " AND " in query
        # Should have Humans filter
        assert "Humans[MeSH]" in query
    
    def test_build_query_with_modifiers(self, builder):
        """Test query with modifier terms."""
        pico = PICOQuery(
            population=["melanoma"],
            intervention=["treatment"],
            modifiers=["stage II"],
        )
        query = builder.build_query(pico)
        
        assert "stage II" in query or "stage 2" in query
    
    def test_numeral_variants(self, builder):
        """Test Roman/Arabic numeral variant generation."""
        variants = builder._get_numeral_variants("stage II")
        assert "stage 2" in variants
        
        variants = builder._get_numeral_variants("type 1")
        assert "type I" in variants
    
    def test_mesh_injection(self, builder):
        """Test that MeSH terms are injected for known diseases."""
        pico = PICOQuery(
            population=["diabetes"],
            intervention=["diagnosis"],
        )
        query = builder.build_query(pico)
        
        assert "Diabetes Mellitus[MeSH]" in query
    
    def test_topic_normalization(self, builder):
        """Test topic term normalization."""
        pico = PICOQuery(
            population=["asthma"],
            intervention=["treatment"],
        )
        query = builder.build_query(pico)
        
        # Should expand treatment to include therapy, management
        assert "therapy" in query.lower() or "treatment" in query.lower()
    
    def test_tiab_field_tags(self, builder):
        """Test that title/abstract field tags are applied."""
        pico = PICOQuery(
            population=["custom disease"],
            intervention=["custom topic"],
        )
        query = builder.build_query(pico)
        
        assert "[tiab]" in query
    
    def test_date_filter(self, builder):
        """Test date range filter."""
        pico = PICOQuery(
            population=["covid"],
            intervention=["treatment"],
            date_range=(2020, 2024),
        )
        query = builder.build_query(pico)
        
        assert "2020" in query
        assert "2024" in query
        assert "[dp]" in query
    
    def test_study_type_filter(self, builder):
        """Test study type publication type filter."""
        pico = PICOQuery(
            population=["hypertension"],
            intervention=["treatment"],
            study_types=["rct", "meta-analysis"],
        )
        query = builder.build_query(pico)
        
        assert "Randomized Controlled Trial[pt]" in query
        assert "Meta-Analysis[pt]" in query
    
    def test_or_within_concepts(self, builder):
        """Test that OR is used within concept blocks."""
        pico = PICOQuery(
            population=["melanoma", "skin cancer"],
            intervention=["treatment"],
        )
        query = builder.build_query(pico)
        
        # Should have OR within population block
        # The structure should be (term1 OR term2) AND ...
        assert " OR " in query
    
    def test_and_between_concepts(self, builder):
        """Test that AND is used between concept blocks."""
        pico = PICOQuery(
            population=["melanoma"],
            intervention=["treatment"],
            modifiers=["stage II"],
        )
        query = builder.build_query(pico)
        
        # Count AND occurrences - should be between each major block
        and_count = query.count(" AND ")
        assert and_count >= 2  # population AND intervention AND modifiers
    
    def test_no_mesh_mode(self):
        """Test query building without MeSH terms."""
        builder = PubMedQueryBuilder(use_mesh=False)
        pico = PICOQuery(
            population=["melanoma"],
            intervention=["treatment"],
        )
        query = builder.build_query(pico)
        
        # Should still work but without MeSH
        assert "melanoma" in query.lower()
        # MeSH term should not be present for disease (but may appear in Humans filter)
        # Check that disease MeSH is not in disease block
        assert "Melanoma[MeSH]" not in query


class TestPICODecomposition:
    """Tests for PICODecomposition schema."""
    
    def test_decomposition_to_pico_query(self):
        """Test converting decomposition to PICOQuery."""
        decomp = PICODecomposition(
            population_terms=["diabetes", "type 2"],
            intervention_terms=["metformin", "treatment"],
            outcome_terms=["glycemic control"],
            requires_recent=True,
        )
        
        pico = decomp.to_pico_query(recent_years=5)
        
        assert pico.population == ["diabetes", "type 2"]
        assert pico.intervention == ["metformin", "treatment"]
        assert pico.date_range is not None


class TestBuildSimpleQuery:
    """Tests for the build_simple_query helper function."""
    
    def test_simple_query(self):
        """Test simple query helper."""
        query = build_simple_query("melanoma", "treatment")
        
        assert "melanoma" in query.lower()
        assert "treatment" in query.lower()
    
    def test_simple_query_with_modifiers(self):
        """Test simple query with modifiers."""
        query = build_simple_query(
            "melanoma",
            "treatment",
            modifiers=["stage II", "advanced"],
        )
        
        assert "stage" in query.lower()
    
    def test_simple_query_with_years(self):
        """Test simple query with year filter."""
        query = build_simple_query(
            "covid",
            "treatment",
            years=3,
        )
        
        assert "[dp]" in query  # Date publication tag
