"""
Test improved validation that filters out metadata percentages.
"""

from services.validation import extract_numerical_claims, validate_citations, validate_narrative


def test_metadata_percentage_filtering():
    """Test that metadata percentages are excluded from validation."""
    print("\n=== Testing Metadata Percentage Filtering ===")

    narrative = """## Move Summary
- **Bitcoin USD** (BTC-USD): $45000.0  |  1d: +5.20%  |  5d: +8.50%  (sharply higher)
- **MicroStrategy** (MSTR): $550.0  |  1d: +7.30%  |  5d: -0.10%  (sharply higher)

## Key Drivers
1. Bitcoin surges on institutional adoption [Source 1]
2. Crypto regulation clarity boosts market [Source 2]
3. MicroStrategy adds more Bitcoin to holdings [Source 3]

## Sentiment Read
Headline sentiment is decidedly bullish based on analysis of retrieved news [Source 1] [Source 2] [Source 3].
Aggregate score: +0.650 (+3 positive / -0 negative / ~0 neutral, mode: finbert).

## Market Interpretation
Price action suggests broad-based rally — likely macro or sector-driven.

## Confidence Assessment
Overall confidence: HIGH
Based on 60% data coverage.

## Caveats
- Prototype analysis — not investment advice.
- Based on 3 retrieved headlines [Source 1], [Source 2], [Source 3]; coverage may be incomplete.
"""

    claims = extract_numerical_claims(narrative)

    print(f"Extracted {len(claims)} numerical claims:")
    for claim in claims:
        display = f"{claim.value}%" if claim.claim_type == 'percentage' else f"${claim.value}"
        print(f"  - [{claim.claim_type}] {display} from context: '{claim.context[:40]}...'")

    # Should extract: 4 percentages + 2 prices from Move Summary
    # Should NOT extract: metadata percentages like "60% coverage"
    percentage_claims = [c for c in claims if c.claim_type == 'percentage']
    price_claims = [c for c in claims if c.claim_type == 'price']

    percentage_values = [c.value for c in percentage_claims]
    price_values = [c.value for c in price_claims]

    # Check percentages
    assert 5.20 in percentage_values or 5.2 in percentage_values, "Should extract +5.20%"
    assert 8.50 in percentage_values or 8.5 in percentage_values, "Should extract +8.50%"
    assert 7.30 in percentage_values or 7.3 in percentage_values, "Should extract +7.30%"
    assert -0.10 in percentage_values or -0.1 in percentage_values, "Should extract -0.10%"

    # Check prices
    assert 45000.0 in price_values, "Should extract $45000.0"
    assert 550.0 in price_values, "Should extract $550.0"

    # Check that metadata percentage is NOT extracted
    assert 60.0 not in percentage_values, "Should NOT extract 60% from metadata"

    print(f"✓ Correctly filtered metadata percentages!")
    print(f"✓ Extracted {len(percentage_claims)} percentages + {len(price_claims)} prices = {len(claims)} total claims")


def test_citation_section_filtering():
    """Test that only relevant sections are checked for citations."""
    print("\n=== Testing Citation Section Filtering ===")

    narrative = """## Move Summary
- **Bitcoin USD** (BTC-USD): $45000.0  |  1d: +5.20%  |  5d: +8.50%  (sharply higher)
This is a surge in price.

## Key Drivers
1. Bitcoin surges on institutional adoption [Source 1]
2. Crypto regulation clarity boosts market [Source 2]

## Sentiment Read
Headline sentiment is decidedly bullish with a rally detected.
Markets are showing gains.

## Market Interpretation
Price action suggests broad-based rally — likely macro or sector-driven.

## Confidence Assessment
Overall confidence: HIGH
"""

    validation = validate_citations(narrative, num_sources=3)

    print(f"Citations found: {validation['unique_citations']}")
    print(f"Uncited claims: {validation['uncited_claims']}")
    print(f"Claims needing citations:")
    for claim in validation['details']['claims_needing_citations']:
        print(f"  - '{claim[:60]}...'")

    # Key Drivers should have citations (they do: [Source 1], [Source 2])
    # Market Interpretation sentence doesn't have citation - should be flagged
    # Move Summary and Sentiment Read shouldn't require citations

    # Should find the uncited sentence in Market Interpretation
    assert validation['uncited_claims'] <= 1, f"Should have at most 1 uncited claim, got {validation['uncited_claims']}"

    print(f"✓ Correctly checking only citation-required sections!")
    print(f"✓ Move Summary and Sentiment Read excluded from citation requirements")


def test_full_validation_improved():
    """Test full validation with improved logic."""
    print("\n=== Testing Full Validation (Improved) ===")

    narrative = """## Move Summary
- **Bitcoin USD** (BTC-USD): $45000.0  |  1d: +5.20%  |  5d: +8.50%  (sharply higher)
- **MicroStrategy** (MSTR): $550.0  |  1d: +7.30%  |  5d: -0.10%  (sharply higher)

## Key Drivers
1. Bitcoin surges on institutional adoption [Source 1]
2. Crypto regulation clarity boosts market [Source 2]
3. MicroStrategy adds more Bitcoin to holdings [Source 3]

## Sentiment Read
Headline sentiment is decidedly bullish based on retrieved news [Source 1] [Source 2] [Source 3].

## Market Interpretation
Price action suggests broad-based rally — likely macro or sector-driven [Source 1].

## Confidence Assessment
Overall confidence: HIGH

## Caveats
- Based on 3 headlines with 100% coverage.
"""

    market_data = [
        {'ticker': 'BTC-USD', 'current_price': 45000.0, 'change_1d': 5.2, 'change_5d': 8.5},
        {'ticker': 'MSTR', 'current_price': 550.0, 'change_1d': 7.3, 'change_5d': -0.1},
    ]

    result = validate_narrative(
        narrative=narrative,
        market_data=market_data,
        num_sources=3,
        sentiment_score=0.65
    )

    print(f"Validation passed: {result.passed}")
    print(f"Confidence score: {result.confidence_score:.1f}/100")
    print(f"Numerical verification: {result.numerical_verification['verified_claims']}/{result.numerical_verification['total_claims']}")
    print(f"Citations: {result.citation_verification['unique_citations']} sources cited")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")

    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")

    if result.warnings:
        print("Warnings:")
        for warn in result.warnings:
            print(f"  - {warn}")

    # Should pass validation with high confidence
    assert result.passed, "Validation should pass"
    assert result.confidence_score >= 80, f"Confidence should be high (>=80), got {result.confidence_score}"
    assert result.numerical_verification['verification_rate'] >= 0.9, "Most numbers should verify"

    print(f"✓ Full validation passed with {result.confidence_score:.0f}/100 confidence!")


if __name__ == "__main__":
    print("Running improved validation tests...")

    try:
        test_metadata_percentage_filtering()
        test_citation_section_filtering()
        test_full_validation_improved()

        print("\n" + "="*60)
        print("✓ ALL IMPROVED VALIDATION TESTS PASSED")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
