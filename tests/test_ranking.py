from datetime import datetime, timezone

from ai_safety_brief.models import CandidateItem, SourceDefinition
from ai_safety_brief.services.ranking import dedupe_candidates, score_candidate


def test_dedupe_candidates_collapses_similar_titles():
    items = [
        CandidateItem(
            source_key="alignment_forum",
            source_name="Alignment Forum",
            title="A roadmap for scalable oversight",
            canonical_url="https://example.org/post-1",
            content_type="opinion",
        ),
        CandidateItem(
            source_key="lesswrong",
            source_name="LessWrong",
            title="A roadmap for scalable oversight ",
            canonical_url="https://example.org/post-2",
            content_type="opinion",
        ),
    ]
    deduped = dedupe_candidates(items)
    assert len(deduped) == 1


def test_score_candidate_rewards_relevance_and_authority():
    source = SourceDefinition(
        key="anthropic_research",
        name="Anthropic Research",
        mode="listing",
        url="https://example.org",
        authority_weight=1.4,
        content_type="news",
    )
    item = CandidateItem(
        source_key=source.key,
        source_name=source.name,
        title="New alignment evaluations for dangerous capability discovery",
        canonical_url="https://example.org/post",
        content_type="news",
        excerpt="The work focuses on alignment evaluations, oversight, and red teaming.",
        published_at=datetime(2026, 4, 8, 6, 0, tzinfo=timezone.utc),
    )
    score = score_candidate(item, source, datetime(2026, 4, 8, 8, 0, tzinfo=timezone.utc))
    assert score > 20

