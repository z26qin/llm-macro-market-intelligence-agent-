"""
Validation service for anti-hallucination features.

This module provides functions to:
1. Verify numerical claims against source data
2. Validate citation presence and correctness
3. Detect sentiment-narrative mismatches
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class NumericalClaim:
    """Represents a numerical claim found in text"""
    value: float
    context: str  # Surrounding text for context
    claim_type: str  # 'percentage' or 'price'
    position: int  # Character position in text


@dataclass
class ValidationResult:
    """Results from validation checks"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    numerical_verification: Dict[str, Any]
    citation_verification: Dict[str, Any]
    confidence_score: float  # Overall confidence (0-100)
    attempts: int = 1  # Number of LLM generation attempts (1 = no self-correction needed)


def extract_numerical_claims(text: str) -> List[NumericalClaim]:
    """
    Extract all numerical claims from narrative text.

    Captures:
    - Percentages: +2.3%, -1.5%, 2.3%
    - Prices: $123.45, $1,234.56
    - Basis points: 50bps, 50 basis points

    Args:
        text: The narrative text to analyze

    Returns:
        List of NumericalClaim objects
    """
    claims = []

    # Split text into sections to avoid extracting metadata percentages
    # Only check Move Summary and Market Interpretation sections for price claims
    lines = text.split('\n')

    # Find relevant sections (avoid Caveats, Confidence sections)
    # Matches both template-style (## Heading) and LLM-style (4. OVERALL CONFIDENCE SCORE) headings
    exclude_sections = ['## Caveats', '## Confidence Assessment', '## Sentiment Read']
    exclude_patterns = ['OVERALL CONFIDENCE SCORE', 'CONFIDENCE SCORE', 'OVERALL CONFIDENCE']
    include_text = []
    skip_section = False

    for line in lines:
        # Check if we're entering an excluded section (template or LLM format)
        if any(line.startswith(section) for section in exclude_sections):
            skip_section = True
            continue
        if any(pattern in line.upper() for pattern in exclude_patterns):
            skip_section = True
            continue
        # Check if we're entering a new section (reset skip)
        if line.startswith('##') or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
            skip_section = False

        if not skip_section:
            include_text.append(line)

    # Only analyze relevant sections for price claims
    relevant_text = '\n'.join(include_text)

    # Pattern for percentages (with optional +/- and %)
    percentage_pattern = r'([+-]?\d+\.?\d*)%'
    for match in re.finditer(percentage_pattern, relevant_text):
        value = float(match.group(1))
        # Get surrounding context (30 chars before and after for better context)
        start = max(0, match.start() - 30)
        end = min(len(relevant_text), match.end() + 30)
        context = relevant_text[start:end]

        # Additional filtering: skip if context suggests it's metadata
        # Skip percentages near words like "coverage", "complete", "based on"
        context_lower = context.lower()
        metadata_keywords = ['coverage', 'complete', 'based on', 'retrieved', 'sources cited']

        if any(keyword in context_lower for keyword in metadata_keywords):
            continue

        claims.append(NumericalClaim(
            value=value,
            context=context,
            claim_type='percentage',
            position=match.start()
        ))

    # Pattern for prices ($) - matches any number of digits with optional decimals
    # Format: $123.45, $1,234.56, $45000.0, etc.
    price_pattern = r'\$(\d{1,}(?:,\d{3})*(?:\.\d+)?)'
    for match in re.finditer(price_pattern, relevant_text):  # Use relevant_text, not full text
        # Remove commas and convert to float
        value_str = match.group(1).replace(',', '')
        value = float(value_str)

        start = max(0, match.start() - 30)
        end = min(len(relevant_text), match.end() + 30)
        context = relevant_text[start:end]

        claims.append(NumericalClaim(
            value=value,
            context=context,
            claim_type='price',
            position=match.start()
        ))

    return claims


def verify_percentage_claim(claimed: float, actual: float, tolerance: float = 0.2) -> bool:
    """
    Check if a claimed percentage is within tolerance of actual value.

    Args:
        claimed: The percentage claimed in narrative
        actual: The actual percentage from data
        tolerance: Acceptable difference (default 0.2 percentage points)

    Returns:
        True if within tolerance, False otherwise
    """
    return abs(claimed - actual) <= tolerance


def verify_numerical_claims(
    narrative: str,
    market_data: List[Dict[str, Any]],
    tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    Verify all numerical claims in narrative against source market data.

    Args:
        narrative: Generated narrative text
        market_data: List of ticker data with prices and changes
        tolerance: Acceptable percentage difference

    Returns:
        Dict with verification results
    """
    claims = extract_numerical_claims(narrative)

    # Build lookup tables from market data
    price_data = {}
    change_1d_data = {}
    change_5d_data = {}

    for ticker_info in market_data:
        ticker = ticker_info['ticker']
        price_data[ticker] = ticker_info.get('current_price')
        change_1d_data[ticker] = ticker_info.get('change_1d')
        change_5d_data[ticker] = ticker_info.get('change_5d')

    verified_claims = []
    unverified_claims = []
    mismatches = []

    for claim in claims:
        verified = False

        if claim.claim_type == 'percentage':
            # Prefer matching against a ticker mentioned in the claim's context window
            context_upper = claim.context.upper()
            context_tickers = [t for t in change_1d_data if t.upper() in context_upper]
            # Fall back to all tickers if none found in context
            tickers_to_check = context_tickers if context_tickers else list(change_1d_data.keys())

            for ticker in tickers_to_check:
                actual_change = change_1d_data.get(ticker)
                if actual_change is not None:
                    if verify_percentage_claim(claim.value, actual_change, tolerance):
                        verified = True
                        verified_claims.append({
                            'claim': claim.value,
                            'actual': actual_change,
                            'ticker': ticker,
                            'context': claim.context
                        })
                        break

            if not verified:
                tickers_to_check_5d = context_tickers if context_tickers else list(change_5d_data.keys())
                for ticker in tickers_to_check_5d:
                    actual_change = change_5d_data.get(ticker)
                    if actual_change is not None:
                        if verify_percentage_claim(claim.value, actual_change, tolerance):
                            verified = True
                            verified_claims.append({
                                'claim': claim.value,
                                'actual': actual_change,
                                'ticker': ticker,
                                'context': claim.context
                            })
                            break

        elif claim.claim_type == 'price':
            # Check against current prices (with 5% tolerance for prices)
            price_tolerance_pct = 5.0
            for ticker, actual_price in price_data.items():
                if actual_price is not None:
                    price_diff_pct = abs((claim.value - actual_price) / actual_price * 100)
                    if price_diff_pct <= price_tolerance_pct:
                        verified = True
                        verified_claims.append({
                            'claim': claim.value,
                            'actual': actual_price,
                            'ticker': ticker,
                            'context': claim.context
                        })
                        break

        if not verified:
            unverified_claims.append({
                'value': claim.value,
                'type': claim.claim_type,
                'context': claim.context
            })
            # Check if this is a clear mismatch (claim doesn't match any data)
            if claim.claim_type == 'percentage':
                all_changes = [v for v in list(change_1d_data.values()) + list(change_5d_data.values()) if v is not None]
                closest = min(all_changes, key=lambda x: abs(x - claim.value)) if all_changes else None
                if closest and abs(closest - claim.value) > 1.0:  # More than 1% off
                    mismatches.append({
                        'claimed': claim.value,
                        'closest_actual': closest,
                        'difference': abs(closest - claim.value),
                        'context': claim.context
                    })

    total_claims = len(claims)
    verification_rate = len(verified_claims) / total_claims if total_claims > 0 else 1.0

    return {
        'total_claims': total_claims,
        'verified_claims': len(verified_claims),
        'unverified_claims': len(unverified_claims),
        'mismatches': len(mismatches),
        'verification_rate': verification_rate,
        'passed': len(mismatches) == 0,  # Pass if no clear mismatches
        'details': {
            'verified': verified_claims,
            'unverified': unverified_claims,
            'mismatches': mismatches
        }
    }


def validate_citations(narrative: str, num_sources: int) -> Dict[str, Any]:
    """
    Validate that citations are present and correctly formatted.

    Expected format: [Source 1], [Source 2], etc.

    Args:
        narrative: Generated narrative text
        num_sources: Number of available sources

    Returns:
        Dict with citation validation results
    """
    # Find all citations in format [Source N]
    citation_pattern = r'\[Source (\d+)\]'
    citations = re.findall(citation_pattern, narrative)

    # Convert to integers
    citation_nums = [int(c) for c in citations]
    unique_citations = set(citation_nums)

    # Check for invalid citations (references to non-existent sources)
    invalid_citations = [c for c in citation_nums if c < 1 or c > num_sources]

    # Extract only sections that should have citations (exclude data reporting sections)
    lines = narrative.split('\n')
    citation_required_sections = []
    current_section = None
    # Sections that should have citations for claims
    require_citations = ['## Key Drivers', '## Market Interpretation']
    # Sections that don't need citations (data reporting or metadata)
    skip_sections = ['## Move Summary', '## Sentiment Read', '## Confidence Assessment', '## Caveats']

    for line in lines:
        if line.startswith('##'):
            current_section = line.strip()
        elif current_section and current_section in require_citations:
            citation_required_sections.append(line)

    citation_required_text = '\n'.join(citation_required_sections)
    sentences = re.split(r'[.!?]+', citation_required_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]

    citation_coverage = len(unique_citations) / num_sources if num_sources > 0 else 0

    # Check if claims in citation-required sections have citations
    # Only check sentences with substantive market claims (not just formatting)
    non_claim_prefixes = (
        'price action suggests', 'note:', 'overall confidence',
        'limiting factors', 'prototype', 'based on', 'headline sentiment',
        'aggregate score', 'insufficient data',
    )
    claims_needing_citations = []
    for sentence in sentences:
        has_citation = bool(re.search(citation_pattern, sentence))

        # Skip empty or very short sentences
        if len(sentence) < 15:
            continue

        # Skip sentences that are just section headers or formatting
        if sentence.startswith('#') or sentence.startswith('-') or sentence.startswith('*'):
            continue

        # Skip interpretive/template sentences that don't need a source citation
        if sentence.lower().startswith(non_claim_prefixes):
            continue

        # If it's a substantive sentence without citation, flag it
        if not has_citation:
            claims_needing_citations.append(sentence.strip())

    passed = len(invalid_citations) == 0 and len(claims_needing_citations) == 0

    warnings = []
    if citation_coverage < 0.5:
        warnings.append(f"Low citation coverage: Only {len(unique_citations)}/{num_sources} sources cited")

    return {
        'total_citations': len(citation_nums),
        'unique_citations': len(unique_citations),
        'invalid_citations': invalid_citations,
        'citation_coverage': citation_coverage,
        'uncited_claims': len(claims_needing_citations),
        'passed': passed,
        'warnings': warnings,
        'details': {
            'claims_needing_citations': claims_needing_citations
        }
    }


def detect_sentiment_mismatch(narrative: str, sentiment_score: float) -> Dict[str, Any]:
    """
    Detect if narrative sentiment contradicts actual headline sentiment.

    Args:
        narrative: Generated narrative text
        sentiment_score: Aggregate sentiment score from FinBERT (-1 to +1)

    Returns:
        Dict with mismatch detection results
    """
    # Define sentiment indicators
    bullish_words = ['surge', 'rally', 'bullish', 'optimistic', 'gains', 'soar', 'jump',
                     'positive', 'strength', 'upbeat', 'favorable', 'encouraging']
    bearish_words = ['crash', 'plunge', 'bearish', 'pessimistic', 'losses', 'tumble',
                     'decline', 'negative', 'weakness', 'downbeat', 'concerning', 'disappointing']

    narrative_lower = narrative.lower()

    # Count sentiment indicators in narrative
    bullish_count = sum(1 for word in bullish_words if word in narrative_lower)
    bearish_count = sum(1 for word in bearish_words if word in narrative_lower)

    # Determine narrative tone
    if bullish_count > bearish_count * 1.5:
        narrative_tone = 'bullish'
    elif bearish_count > bullish_count * 1.5:
        narrative_tone = 'bearish'
    else:
        narrative_tone = 'neutral'

    # Determine actual sentiment tone
    if sentiment_score > 0.2:
        actual_tone = 'bullish'
    elif sentiment_score < -0.2:
        actual_tone = 'bearish'
    else:
        actual_tone = 'neutral'

    # Check for mismatch
    mismatch = False
    mismatch_type = None

    if narrative_tone == 'bullish' and actual_tone == 'bearish':
        mismatch = True
        mismatch_type = 'bullish_narrative_bearish_data'
    elif narrative_tone == 'bearish' and actual_tone == 'bullish':
        mismatch = True
        mismatch_type = 'bearish_narrative_bullish_data'

    return {
        'mismatch': mismatch,
        'mismatch_type': mismatch_type,
        'narrative_tone': narrative_tone,
        'actual_tone': actual_tone,
        'sentiment_score': sentiment_score,
        'bullish_indicators': bullish_count,
        'bearish_indicators': bearish_count
    }


def calculate_confidence_score(
    numerical_verification: Dict[str, Any],
    citation_verification: Dict[str, Any],
    sentiment_check: Dict[str, Any]
) -> float:
    """
    Calculate overall confidence score (0-100) based on validation results.

    Args:
        numerical_verification: Results from verify_numerical_claims
        citation_verification: Results from validate_citations
        sentiment_check: Results from detect_sentiment_mismatch

    Returns:
        Confidence score 0-100
    """
    score = 100.0

    # Penalty for numerical mismatches (severe)
    if numerical_verification['mismatches'] > 0:
        score -= numerical_verification['mismatches'] * 20

    # Penalty for unverified claims (moderate)
    if numerical_verification['total_claims'] > 0:
        unverified_rate = numerical_verification['unverified_claims'] / numerical_verification['total_claims']
        score -= unverified_rate * 15

    # Penalty for missing citations (moderate)
    if citation_verification['uncited_claims'] > 0:
        score -= citation_verification['uncited_claims'] * 10

    # Penalty for invalid citations (severe)
    score -= len(citation_verification['invalid_citations']) * 25

    # Penalty for sentiment mismatch (moderate)
    if sentiment_check['mismatch']:
        score -= 20

    # Bonus for good citation coverage
    if citation_verification['citation_coverage'] >= 0.8:
        score += 5

    return max(0.0, min(100.0, score))


def validate_narrative(
    narrative: str,
    market_data: List[Dict[str, Any]],
    num_sources: int,
    sentiment_score: float
) -> ValidationResult:
    """
    Comprehensive validation of generated narrative.

    Args:
        narrative: Generated narrative text
        market_data: Market data used for generation
        num_sources: Number of sources available
        sentiment_score: Aggregate sentiment score

    Returns:
        ValidationResult object with all validation checks
    """
    # Run all validation checks
    numerical_verification = verify_numerical_claims(narrative, market_data)
    citation_verification = validate_citations(narrative, num_sources)
    sentiment_check = detect_sentiment_mismatch(narrative, sentiment_score)

    # Calculate overall confidence
    confidence = calculate_confidence_score(
        numerical_verification,
        citation_verification,
        sentiment_check
    )

    # Collect errors and warnings
    errors = []
    warnings = []

    # Numerical verification errors
    if numerical_verification['mismatches'] > 0:
        for mismatch in numerical_verification['details']['mismatches']:
            errors.append(
                f"Numerical mismatch: Claimed {mismatch['claimed']}% but closest actual is {mismatch['closest_actual']:.2f}%"
            )

    # Citation errors
    if citation_verification['invalid_citations']:
        errors.append(
            f"Invalid citations to non-existent sources: {citation_verification['invalid_citations']}"
        )

    if citation_verification['uncited_claims'] > 0:
        warnings.append(
            f"{citation_verification['uncited_claims']} factual claims lack citations"
        )

    # Citation warnings
    warnings.extend(citation_verification['warnings'])

    # Sentiment mismatch warning
    if sentiment_check['mismatch']:
        warnings.append(
            f"Sentiment mismatch: Narrative is {sentiment_check['narrative_tone']} but data is {sentiment_check['actual_tone']}"
        )

    passed = len(errors) == 0

    return ValidationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        numerical_verification=numerical_verification,
        citation_verification=citation_verification,
        confidence_score=confidence
    )
