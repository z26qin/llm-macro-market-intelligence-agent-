"""
Test script for validation module.

Tests the anti-hallucination features:
1. Numerical claim extraction and verification
2. Citation validation
3. Sentiment-narrative mismatch detection
"""

from services.validation import (
    extract_numerical_claims,
    verify_numerical_claims,
    validate_citations,
    detect_sentiment_mismatch,
    validate_narrative
)


def test_numerical_extraction():
    """Test extraction of numbers from text."""
    print("\n=== Testing Numerical Extraction ===")

    text = """NVDA surged +5.2% today while MSFT dropped -2.1%.
    The stock is trading at $145.50 after the rally."""

    claims = extract_numerical_claims(text)
    print(f"Found {len(claims)} numerical claims:")
    for claim in claims:
        print(f"  - {claim.claim_type}: {claim.value} in context: '{claim.context}'")

    assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"
    print("✓ Numerical extraction test passed")


def test_numerical_verification():
    """Test verification of numerical claims against market data."""
    print("\n=== Testing Numerical Verification ===")

    narrative = """NVDA gained +5.0% today while MSFT fell -2.0%.
    The overall market was mixed with these divergent moves."""

    market_data = [
        {'ticker': 'NVDA', 'current_price': 140.0, 'change_1d': 5.1, 'change_5d': 8.2},
        {'ticker': 'MSFT', 'current_price': 420.0, 'change_1d': -2.1, 'change_5d': -1.0},
    ]

    result = verify_numerical_claims(narrative, market_data, tolerance=0.2)
    print(f"Verification rate: {result['verification_rate']*100:.0f}%")
    print(f"Verified claims: {result['verified_claims']}/{result['total_claims']}")
    print(f"Mismatches: {result['mismatches']}")

    assert result['verification_rate'] >= 0.9, "Verification rate should be high for accurate numbers"
    print("✓ Numerical verification test passed")


def test_citation_validation():
    """Test citation validation."""
    print("\n=== Testing Citation Validation ===")

    # Good narrative with citations
    good_narrative = """NVDA surged +5.2% [Source 1] driven by strong earnings [Source 2].
    Analysts upgraded the stock [Source 3]."""

    result = validate_citations(good_narrative, num_sources=5)
    print(f"Good narrative - Citations: {result['unique_citations']}, Coverage: {result['citation_coverage']*100:.0f}%")

    # Bad narrative without citations
    bad_narrative = """NVDA surged +5.2% driven by strong earnings.
    Analysts upgraded the stock."""

    result_bad = validate_citations(bad_narrative, num_sources=5)
    print(f"Bad narrative - Uncited claims: {result_bad['uncited_claims']}")

    assert result['unique_citations'] == 3, "Should find 3 unique citations"
    assert result_bad['uncited_claims'] > 0, "Should detect uncited claims"
    print("✓ Citation validation test passed")


def test_sentiment_mismatch():
    """Test sentiment-narrative mismatch detection."""
    print("\n=== Testing Sentiment Mismatch Detection ===")

    # Bullish narrative with bearish sentiment
    bullish_narrative = """Markets surged today with strong gains across all sectors.
    Investors were optimistic and bullish."""

    result = detect_sentiment_mismatch(bullish_narrative, sentiment_score=-0.5)
    print(f"Bullish narrative + bearish data - Mismatch: {result['mismatch']}, Type: {result['mismatch_type']}")

    assert result['mismatch'] == True, "Should detect mismatch"
    assert result['mismatch_type'] == 'bullish_narrative_bearish_data', "Should identify correct mismatch type"

    # Aligned narrative and sentiment
    aligned_result = detect_sentiment_mismatch(bullish_narrative, sentiment_score=0.6)
    print(f"Aligned narrative + data - Mismatch: {aligned_result['mismatch']}")

    assert aligned_result['mismatch'] == False, "Should not detect mismatch when aligned"
    print("✓ Sentiment mismatch detection test passed")


def test_full_validation():
    """Test complete validation workflow."""
    print("\n=== Testing Full Validation ===")

    narrative = """NVDA gained +5.1% today [Source 1] on strong earnings results [Source 2].
    The company reported record revenue [Source 3], driving bullish sentiment across tech stocks."""

    market_data = [
        {'ticker': 'NVDA', 'current_price': 140.0, 'change_1d': 5.1, 'change_5d': 8.2},
    ]

    result = validate_narrative(
        narrative=narrative,
        market_data=market_data,
        num_sources=5,
        sentiment_score=0.6  # Bullish sentiment matches bullish narrative
    )

    print(f"Validation passed: {result.passed}")
    print(f"Confidence score: {result.confidence_score:.1f}/100")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")

    if result.errors:
        print("Errors found:")
        for err in result.errors:
            print(f"  - {err}")

    if result.warnings:
        print("Warnings:")
        for warn in result.warnings:
            print(f"  - {warn}")

    assert result.passed, "Validation should pass for well-cited narrative with accurate numbers"
    assert result.confidence_score >= 70, "Confidence should be reasonably high"
    print("✓ Full validation test passed")


if __name__ == "__main__":
    print("Running validation module tests...")

    try:
        test_numerical_extraction()
        test_numerical_verification()
        test_citation_validation()
        test_sentiment_mismatch()
        test_full_validation()

        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED")
        print("="*50)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
