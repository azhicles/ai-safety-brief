from ai_safety_brief.models import CandidateItem
from ai_safety_brief.services.summarizer import Summarizer


def test_summarizer_extracts_high_signal_sentences():
    item = CandidateItem(
        source_key="metr_blog",
        source_name="METR",
        title="New oversight benchmark for autonomous coding agents",
        canonical_url="https://example.org/post",
        content_type="news",
        excerpt="Researchers tested whether autonomous coding agents can hide harmful actions from monitors.",
        raw_text=(
            "Researchers tested whether autonomous coding agents can hide harmful actions from monitors. "
            "The benchmark measures oversight failure under realistic pressure. "
            "The team argues this helps labs evaluate sabotage risk before deployment."
        ),
    )
    summary, why = Summarizer().summarize(item)
    assert "autonomous coding agents" in summary.lower()
    assert "oversight" in why.lower() or "detect" in why.lower()

