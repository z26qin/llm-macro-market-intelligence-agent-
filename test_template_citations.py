"""
Test that the template generator now includes citations.
"""

from services.narrative import generate_narrative
from services.search import SearchResult
from services.market_data import PriceSnapshot
from services.sentiment import SentimentSummary, SentimentResult
from services.validation import validate_citations


def test_template_includes_citations():
    """Verify template generator includes [Source N] citations."""
    print("\n=== Testing Template Generator Citations ===")

    # Mock data
    results = [
        SearchResult(
            title="Bitcoin surges on institutional adoption",
            url="https://example.com/1",
            snippet="Major institutions are buying crypto",
            published_date="2025-01-15T10:00:00Z"
        ),
        SearchResult(
            title="Crypto regulation clarity boosts market",
            url="https://example.com/2",
            snippet="New regulations provide certainty",
            published_date="2025-01-15T09:00:00Z"
        ),
        SearchResult(
            title="MicroStrategy adds more Bitcoin to holdings",
            url="https://example.com/3",
            snippet="Company continues accumulation strategy",
            published_date="2025-01-15T08:00:00Z"
        ),
    ]

    snapshots = [
        PriceSnapshot(
            ticker="BTC-USD",
            name="Bitcoin USD",
            price=45000.0,
            change_1d_pct=5.2,
            change_5d_pct=8.5,
            error=None
        ),
        PriceSnapshot(
            ticker="MSTR",
            name="MicroStrategy",
            price=550.0,
            change_1d_pct=7.3,
            change_5d_pct=12.1,
            error=None
        ),
    ]

    sentiment = SentimentSummary(
        avg_score=0.65,
        positive=3,
        negative=0,
        neutral=0,
        mode="finbert",
        details=[
            SentimentResult(label="positive", score=0.85, text="Bitcoin surges..."),
            SentimentResult(label="positive", score=0.75, text="Crypto regulation..."),
            SentimentResult(label="positive", score=0.55, text="MicroStrategy adds..."),
        ]
    )

    # Generate narrative
    narrative = generate_narrative(
        query="crypto",
        query_type="crypto",
        results=results,
        snapshots=snapshots,
        sentiment=sentiment
    )

    print("Generated narrative:")
    print("=" * 60)
    print(narrative)
    print("=" * 60)

    # Validate citations
    validation = validate_citations(narrative, num_sources=len(results))

    print(f"\nCitation validation:")
    print(f"  Total citations: {validation['total_citations']}")
    print(f"  Unique sources cited: {validation['unique_citations']}")
    print(f"  Citation coverage: {validation['citation_coverage']*100:.0f}%")
    print(f"  Uncited claims: {validation['uncited_claims']}")
    print(f"  Passed: {validation['passed']}")

    # Check that citations are present
    assert "[Source 1]" in narrative, "Should contain [Source 1]"
    assert "[Source 2]" in narrative, "Should contain [Source 2]"
    assert "[Source 3]" in narrative, "Should contain [Source 3]"

    # Check that citation coverage is reasonable
    assert validation['unique_citations'] >= 3, f"Should cite at least 3 sources, got {validation['unique_citations']}"
    assert validation['citation_coverage'] >= 0.8, f"Citation coverage should be high, got {validation['citation_coverage']*100:.0f}%"

    # Check that confidence section is present
    assert "## Confidence Assessment" in narrative, "Should include confidence assessment"
    assert "confidence:" in narrative.lower(), "Should include confidence rating"

    print("\n✓ Template generator now includes proper citations!")
    print(f"✓ Citation coverage: {validation['citation_coverage']*100:.0f}%")


if __name__ == "__main__":
    try:
        test_template_includes_citations()
        print("\n" + "=" * 60)
        print("✓ TEST PASSED - Template citations working!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
