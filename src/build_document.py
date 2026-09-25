#!/usr/bin/env python3
"""Render a two-page methodology, clearly marking deferred evaluation as draft."""
import argparse
from datetime import date
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]


def build(team, members, report_path, destination):
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    status = "Measured training holdout; test leaderboard score unknown" if report else "DRAFT - FULL DATA EVALUATION NOT YET RUN"
    identity = f"Project/team: {team} | Members: {members or 'Not yet supplied'} | Date: {date.today().isoformat()}"
    sections = [
        ("1. Executive summary", "EntityMatch uses country-aware indexed blocking followed by a compact CatBoost classifier. It resolves every Source 1 reference against Source 2 and Source 3 while limiting the number of pairs sent to the classifier. All identity evidence comes from the provided files; there are no external lookups, geocoding calls, or pretrained model downloads."),
        ("2. Methodology", "The supplied files contain 2,206,821 training references and 10,320,219 training targets, plus 1,732,544 test references and 9,969,589 test targets. Names and addresses contain abbreviations, reordered words, spelling noise, and missing fields. The implementation folds common Latin accents, removes common legal suffixes, and normalizes common address abbreviations. Country is an open string label; test-only France is retained. IDs are used for joining and deterministic splitting, never as classifier features."),
        ("3. Candidate generation", "A sorted 64-bit posting index covers normalized names, sorted name tokens, full addresses, and name-token prefixes combined with address-word prefixes or numbers. Keys appearing in more than 64 reference records are excluded before training sampling. Targets are streamed through the index, with an explicit country equality check. Pairs with name Dice below 0.12 or weighted name/address similarity below 0.27 are removed. The final deterministic blocking stage retains at most 12 candidates per reference, ranked by 0.65 name Dice plus 0.35 address Dice. These defaults require full-data validation. Every retained pair is sent to the classifier and exported in candidate_pairs.tsv."),
        ("4. Matching model", "The classifier uses 22 features: character-bigram Dice, token Jaccard and containment, token-sorted name similarity, exact agreement, length ratios, first-token/number agreement, numeric overlap/conflict, missingness flags, and name/address interactions. CatBoost uses depth 6, learning rate 0.06, a fixed seed, and up to 600 trees with early stopping. It is trained from scratch with an Apache-2.0 implementation; the project provides its original code and trained models under MIT. The model is far below the eight-billion-parameter limit."),
        ("5. Evaluation and error analysis", "Training defaults to a deterministic 2% reference sample, with the complete target corpus available for retrieval. A separate hash partitions references into 60% fitting, 20% tuning, and 20% holdout groups. The tuning set selects tree count and a probability threshold maximizing entity-macro F0.5. The holdout remains unused for fitting or threshold selection. Unretrieved true links count as false negatives, and correctly empty singletons score 1. The final model is refit on all sampled training labels after evaluation."),
        ("6. Results and limitations", ""),
        ("7. Reproduction and submission", "src/pipeline.py provides train, predict, validate, and package commands; src/retrieve.cpp supplies native retrieval and features. The README gives exact commands and dependencies are pinned. Prediction writes one ordered row per reference to both TSVs. Validation checks complete coverage, unique IDs, source prefixes, real target-ID membership, and final-match containment in the scored candidate set. Packaging revalidates outputs and creates the required output/ and code/business_entity_resolution/ layout with this methodology. The raw dataset is supplied separately."),
        ("8. Conclusion", "The implementation prioritizes compact candidate sets and precision-sensitive classification while preserving empty results and unseen-country coverage. Synthetic tests exercise the code and file contracts. Before a competition submission, run the supplied dataset, inspect measured blocking recall and matching errors, finalize team details, and regenerate the methodology and ZIP. Full-data memory, runtime, and France generalization remain to be assessed."),
    ]
    if report:
        h = report["holdout"]
        sections[5] = (sections[5][0], f"Held-out macro F0.5: {h['macro_f0_5']:.6f}; pair precision: {h['pair_precision']:.6f}; pair recall: {h['pair_recall']:.6f}; blocking pair recall: {h['blocking_pair_recall']:.6f}. Sampled candidate pairs: {report['candidate_pairs']:,}; mean holdout candidates per entity: {h['average_candidates']:.3f}; threshold: {report['threshold']:.4f}. These are local training-holdout measurements, not leaderboard results. Review reports/evaluation.json for country breakdowns and error examples. Country filtering, block caps, heavy corruption, and top-k truncation can lose true matches; common names/addresses can create false merges. France has no labeled training examples.")
    else:
        sections[5] = (sections[5][0], "NOT YET RUN: no competition holdout score, candidate recall, runtime, or leaderboard result is available. Full-data execution was deferred at the user's request after a low-disk-space warning. Synthetic tests establish functional behavior only. Likely failure modes include true links lost through heavy corruption, country disagreements, frequent-key exclusion or top-k truncation, and false merges for shared names/addresses. France has no labeled training examples, so France accuracy cannot be inferred from a US/India holdout.")
    if report:
        replacements = {
            "more than 64": f"more than {report['block_cap']}",
            "at most 12": f"at most {report['top_k']}",
            "up to 600 trees": f"{report['iterations']} trees selected from the configured training budget",
            "deterministic 2%": f"deterministic {100 * report['sample_keep'] / report['sample_mod']:g}%",
        }
        for old, new in replacements.items():
            sections = [(title, body.replace(old, new)) for title, body in sections]
    markdown = ["# ML Challenge 2026: Business Entity Resolution", "", f"**Status:** {status}", "", identity, ""]
    for title, body in sections:
        markdown.extend([f"## {title}", "", body, ""])
    (ROOT / "Documentation_template.md").write_text("\n".join(markdown), encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCustom", fontName="Helvetica", fontSize=9.4, leading=13, spaceAfter=9, textColor=colors.HexColor("#243347")))
    styles.add(ParagraphStyle(name="SectionCustom", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=7, spaceAfter=5, textColor=colors.HexColor("#143e63")))
    styles.add(ParagraphStyle(name="SmallCustom", fontName="Helvetica", fontSize=8, leading=11, spaceAfter=8, textColor=colors.HexColor("#536276")))
    story = [Paragraph("ENTITYMATCH", styles["Title"]), Paragraph("Business Entity Resolution | Approach", styles["Heading2"]), Paragraph(html.escape(status), styles["SmallCustom"]), Paragraph(html.escape(identity), styles["SmallCustom"])]
    for i, (title, body) in enumerate(sections):
        if i == 4:
            story.append(PageBreak())
        story.append(KeepTogether([Paragraph(html.escape(title), styles["SectionCustom"]), Paragraph(html.escape(body), styles["BodyCustom"])]))
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#dce3eb"))
        canvas.line(42, 36, 553, 36)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536276"))
        canvas.drawString(42, 24, "EntityMatch | Provided-data-only pipeline")
        canvas.drawRightString(553, 24, f"{doc.page}")
    SimpleDocTemplate(str(destination), pagesize=(595.28, 841.89), rightMargin=42, leftMargin=42,
                      topMargin=34, bottomMargin=48, title="EntityMatch approach", author=team).build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--team", default="EntityMatch (registration details pending)")
    p.add_argument("--members", default="")
    p.add_argument("--report", type=Path, default=ROOT / "reports/evaluation.json")
    p.add_argument("--output", type=Path, default=ROOT / "output/pdf/Approach.pdf")
    args = p.parse_args()
    build(args.team, args.members, args.report, args.output)
