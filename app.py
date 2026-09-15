from __future__ import annotations
import json
import hashlib
import math
import mimetypes
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.config import (
    AVAILABLE_MODELS,
    OUTPUTS_DIR,
    REFERENCE_DOCS_DIR,
    settings,
)
from src.core.pdf_loader import extract_pdf_text, load_pdfs_from_folder
from src.core.rag_engine import RagEngine
from src.evaluation.baseline_gain import (
    gain_vs_baseline,
    interpret_baseline_gain,
)
from src.evaluation.classification_explainer import explain_classification
from src.evaluation.configuration_explainer import explain_best_configuration
from src.evaluation.scientific_summary import write_scientific_summary
from src.evaluation.section_classifier import export_section_classification
from src.experiments.repetition_analyzer import analyze_repetitions, repetition_warning
from src.llm.providers import list_ollama_models
from src.llm.fine_tuned_provider import fine_tuning_readiness
from src.pipeline.experiment_runner import CONFIGURATIONS, ExperimentRunner, results_to_dataframe
from src.pipeline.exporter import (
    evaluation_criteria_dataframe,
    extractions_detailed_dataframe,
    export_detailed_csv,
    write_run_metadata,
    zip_run_directory,
)
from src.pipeline.expert_validation import (
    build_expert_validation_frame,
    save_expert_validation,
)
from src.graph.graph_generator import save_3d_graph, save_graph_data, save_pyvis_graph
from src.pipeline.report_writer import write_global_run_report, write_markdown_report
from src.pipeline.sysml_generator import generate_sysml_v2, save_sysml, validate_sysml_text
from src.rules.rule_repository import load_official_rules_repository
from src.sysml.sysml_graph_generator import save_sysml_graph
from src.utils.io import save_json


st.set_page_config(page_title="ConOps Platform", layout="wide")
UML_DIR = Path(__file__).resolve().parent / "docs" / "uml" / "rendered"
UML_DIAGRAMS = [
    ("Architecture globale", "01_component_architecture.png"),
    ("Sequence C5", "02_sequence_analysis_c5.png"),
    ("Diagramme de classes", "03_class_architecture.png"),
    ("Pipeline experimental", "04_activity_experimental_pipeline.png"),
    (
        "Modele conceptuel ConOps/OpsCon",
        "05_knowledge_model_conops_opscon.png",
    ),
    ("Deploiement local", "06_deployment_architecture.png"),
    ("Pipeline fine-tuning prepare", "07_finetuning_pipeline.png"),
]
st.title("Plateforme ConOps/OpsCon : Evaluation -> SysML v2 -> Graphe")
st.caption(
    "Pipeline experimental : LLM, RAG, metriques, comparaison de configurations "
    "et preparation fine-tuning."
)
st.caption("Rules version: ConOps & OpsCon Evaluation Rules V3.0")
ENABLE_MULTI_DOCUMENT_ANALYSIS = False


@st.cache_data(ttl=15, show_spinner=False)
def detected_ollama_models() -> list[str]:
    return list_ollama_models()


DEFAULT_MODELS = {
    "C1": settings.ollama_model_c1,
    "C2": settings.ollama_model_c2,
    "C3": settings.ollama_model_c3,
    "C4": settings.ollama_model_c4,
    "C5": settings.ollama_model_c5,
    "C6": settings.ollama_model_c6,
    "C7": settings.ollama_model_c7,
}


def dataframe_for_display(data) -> pd.DataFrame:
    """Convert nested cells to JSON strings accepted by Streamlit/Arrow."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        try:
            frame = pd.DataFrame(data)
        except ValueError:
            frame = pd.DataFrame([data])
    else:
        frame = pd.DataFrame(data)

    def normalize_cell(value):
        if isinstance(value, (list, tuple, set, dict)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, bytearray):
            return bytes(value).decode("utf-8", errors="replace")
        return value

    for column in frame.columns:
        if frame[column].dtype != "object":
            continue
        frame[column] = frame[column].map(normalize_cell)
        non_null = frame[column].dropna()
        value_types = {type(value) for value in non_null}
        if len(value_types) > 1:
            frame[column] = frame[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    return frame


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def extract_uploaded_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        temporary_path = Path(temporary_file.name)
    try:
        return extract_pdf_text(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def init_run_session_state() -> None:
    defaults = {
        "last_result": None,
        "last_run_id": "",
        "last_run_dir": "",
        "last_output_dir": "",
        "last_results": None,
        "generated_files": [],
        "analysis_done": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type:
        return mime_type
    suffixes = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".sysml": "text/plain",
        ".html": "text/html",
        ".zip": "application/zip",
        ".txt": "text/plain",
    }
    return suffixes.get(path.suffix.lower(), "application/octet-stream")


def collect_generated_files(output_dir: Path) -> list[dict[str, str]]:
    if not output_dir.exists():
        return []
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(output_dir).as_posix()
        files.append(
            {
                "filename": path.name,
                "relative_path": relative_path,
                "mime_type": guess_mime_type(path),
            }
        )
    return files


def enrich_experimental_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "rule_score_consistency" not in enriched.columns:
        conops = pd.to_numeric(enriched.get("conops_rule_score"), errors="coerce")
        opscon = pd.to_numeric(enriched.get("opscon_rule_score"), errors="coerce")
        enriched["rule_score_consistency"] = (
            100.0 - (conops.fillna(0.0) - opscon.fillna(0.0)).abs()
        ).clip(lower=0.0, upper=100.0)
    if "extraction_consistency_score" not in enriched.columns:
        stability = pd.to_numeric(enriched.get("stability_score"), errors="coerce")
        enriched["extraction_consistency_score"] = stability.fillna(
            100.0 if len(enriched) else 0.0
        )
    if "json_validity_rate" not in enriched.columns:
        json_valid = enriched.get("json_valid")
        if json_valid is not None:
            enriched["json_validity_rate"] = json_valid.map(
                lambda value: 1.0 if bool(value) else 0.0
            )
        else:
            enriched["json_validity_rate"] = 0.0
    return enriched


def write_enriched_summary(run_dir: Path, frame: pd.DataFrame) -> Path:
    summary_path = run_dir / "experiments_summary.csv"
    frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def persist_last_run(
    run_id: str,
    output_dir: Path,
    results: pd.DataFrame | list | dict,
    generated_files: list[dict[str, str]],
) -> None:
    last_result = results.to_dict(orient="records") if isinstance(results, pd.DataFrame) else results
    st.session_state["last_result"] = last_result
    st.session_state["last_run_id"] = run_id
    st.session_state["last_run_dir"] = str(output_dir)
    st.session_state["last_output_dir"] = str(output_dir)
    if isinstance(results, pd.DataFrame):
        st.session_state["last_results"] = results.to_dict(orient="records")
    else:
        st.session_state["last_results"] = results
    st.session_state["generated_files"] = generated_files
    st.session_state["analysis_done"] = True


def latest_output_run_dir() -> Path | None:
    if not OUTPUTS_DIR.exists():
        return None
    run_dirs = [
        path
        for path in OUTPUTS_DIR.glob("run_*")
        if path.is_dir()
        and ((path / "metadata.json").exists() or (path / "run_metadata.json").exists())
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def reload_last_run_from_disk() -> bool:
    run_dir = latest_output_run_dir()
    if run_dir is None:
        return False
    metadata_path = (
        run_dir / "metadata.json"
        if (run_dir / "metadata.json").exists()
        else run_dir / "run_metadata.json"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        metadata = {}
    run_id = metadata.get("run_id") or run_dir.name.removeprefix("run_")
    results_path = run_dir / "experiments_summary.csv"
    if results_path.exists():
        last_results = pd.read_csv(results_path).to_dict(orient="records")
    else:
        last_results = metadata
    persist_last_run(
        run_id=run_id,
        output_dir=run_dir,
        results=last_results,
        generated_files=collect_generated_files(run_dir),
    )
    return True


def render_reload_last_run_option() -> None:
    if st.session_state.get("analysis_done"):
        return
    if latest_output_run_dir() is None:
        return
    if st.button("Recharger dernier run", key="reload_last_run"):
        if reload_last_run_from_disk():
            st.success("Dernier run recharge depuis outputs/.")
        else:
            st.warning("Aucun run precedent rechargeable trouve dans outputs/.")


def render_download_button(
    path: Path,
    run_id: str,
    scope: str,
    index: int,
    label: str | None = None,
) -> None:
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    key_source = "|".join(
        [
            scope,
            run_id,
            str(index),
            str(resolved_path),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ]
    )
    key_hash = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:24]
    with open(path, "rb") as file:
        file_bytes = file.read()
    download_kwargs = {
        "label": label or f"Telecharger {path.name}",
        "data": file_bytes,
        "file_name": path.name,
        "mime": guess_mime_type(path),
        "key": f"download_{run_id}_{index}_{path.name}_{key_hash}",
    }
    st.download_button(**download_kwargs)


FILE_CATEGORIES = [
    (
        "Résumés scientifiques",
        {
            "report_global.md",
            "scientific_summary.md",
            "scientific_report.md",
            "report.json",
        },
    ),
    (
        "Résultats expérimentaux",
        {
            "experiments_summary.csv",
            "evaluation_criteria.csv",
            "ablation_analysis.csv",
            "multi_document_config_stats.csv",
        },
    ),
    (
        "Extraction",
        {
            "extraction_items_detailed.csv",
            "extractions_detailed.csv",
            "analysis_C1.json",
            "analysis_C3.json",
            "analysis_C4.json",
            "analysis_C5.json",
        },
    ),
    ("RAG", {"rag_chunks_used.csv", "rag_rule_traceability.csv"}),
    (
        "Règles officielles",
        {"rule_assessment_detailed.csv", "rule_compliance_matrix.csv"},
    ),
    (
        "Graphes",
        {
            "graph_2d.html",
            "graph_3d.html",
            "graph_data.json",
            "graph_metrics.json",
            "typed_traceability_edges.csv",
        },
    ),
    (
        "SysML",
        {
            "model.sysml",
            "model_C1.sysml",
            "model_C3.sysml",
            "model_C4.sysml",
            "model_C5.sysml",
            "sysml_graph.html",
            "sysml_traceability_matrix.csv",
        },
    ),
    ("Stabilité", {"stability_report.json", "stability_summary.csv"}),
    ("Run complet", set()),
]


def category_for_file(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        return "Run complet"
    for category, filenames in FILE_CATEGORIES:
        if path.name in filenames:
            return category
    return "Autres fichiers"


def render_generated_files_panel() -> None:
    output_dir_value = (
        st.session_state.get("last_run_dir")
        or st.session_state.get("last_output_dir")
    )
    run_id = st.session_state.get("last_run_id", "")
    if not output_dir_value:
        st.info("Les résultats apparaissent après une analyse.")
        return

    output_dir = Path(output_dir_value)
    if not output_dir.exists():
        st.warning(f"Dossier du dernier run introuvable : {output_dir}")
        return

    stored_files = st.session_state.get("generated_files") or []
    files_by_relative_path = {}
    seen: set[Path] = set()
    for file_info in stored_files:
        relative_path = file_info.get("relative_path") or file_info.get("filename")
        if not relative_path:
            continue
        file_path = (output_dir / relative_path).resolve()
        if file_path in seen:
            continue
        seen.add(file_path)
        files_by_relative_path[relative_path] = file_info
    for file_info in collect_generated_files(output_dir):
        file_path = (output_dir / file_info["relative_path"]).resolve()
        if file_path in seen:
            continue
        seen.add(file_path)
        files_by_relative_path[file_info["relative_path"]] = file_info

    if not files_by_relative_path:
        st.info("Aucun fichier généré trouvé pour ce run.")
        return

    zip_files = [
        (relative_path, file_info)
        for relative_path, file_info in files_by_relative_path.items()
        if Path(relative_path).suffix.lower() == ".zip"
    ]
    rendered_paths: set[Path] = set()
    for index, (relative_path, file_info) in enumerate(zip_files):
        file_path = output_dir / relative_path
        if file_path.exists():
            render_download_button(
                path=file_path,
                run_id=run_id,
                scope="run_zip_primary",
                index=index,
                label="Télécharger tout le run ZIP",
            )
            rendered_paths.add(file_path.resolve())
            break

    grouped: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for relative_path, file_info in sorted(files_by_relative_path.items()):
        grouped.setdefault(category_for_file(Path(relative_path)), []).append(
            (relative_path, file_info)
        )

    st.markdown("### Fichiers générés")
    categories = [category for category, _ in FILE_CATEGORIES] + ["Autres fichiers"]
    button_index = 0
    for category in categories:
        entries = grouped.get(category, [])
        if not entries:
            continue
        st.markdown(f"#### {category}")
        for relative_path, _file_info in entries:
            file_path = output_dir / relative_path
            resolved_path = file_path.resolve()
            if resolved_path in rendered_paths:
                continue
            if not file_path.exists():
                st.warning(f"Fichier manquant : {relative_path}")
                continue
            render_download_button(
                path=file_path,
                run_id=run_id,
                scope=category,
                index=button_index,
            )
            rendered_paths.add(resolved_path)
            button_index += 1


def render_experimental_protocol_panel(
    *,
    document_name: str,
    dataframe: pd.DataFrame,
    n_repeats: int,
    model_name: str,
    rag: RagEngine,
    rules_version: str,
) -> None:
    st.markdown("### Protocole expérimental")
    protocol_rows = build_protocol_rows(
        document_name=document_name,
        dataframe=dataframe,
        n_repeats=n_repeats,
        model_name=model_name,
        rag=rag,
        rules_version=rules_version,
    )
    st.dataframe(
        dataframe_for_display(
            pd.DataFrame(protocol_rows, columns=["Paramètre", "Valeur"])
        ),
        use_container_width=True,
    )
    if n_repeats < 5:
        st.warning(
            "Résultat préliminaire : une seule répétition ne suffit pas pour une conclusion scientifique robuste."
        )


def build_protocol_rows(
    *,
    document_name: str,
    dataframe: pd.DataFrame,
    n_repeats: int,
    model_name: str,
    rag: RagEngine,
    rules_version: str,
) -> list[tuple[str, object]]:
    return [
        ("Document analysé", document_name),
        ("Nombre de configurations", int(dataframe["configuration"].nunique())),
        ("Nombre de répétitions", n_repeats),
        ("Modèle LLM", model_name),
        ("Température", 0.0),
        ("RAG interne document analysé", "preuves extraites du PDF cible"),
        ("RAG externe documents de référence", rag.backend),
        ("Embedding model", settings.embedding_model),
        ("Top-k", settings.rag_top_k),
        ("Nombre de chunks indexés", rag.nb_chunks),
        ("Version des règles officielles", rules_version),
    ]


def _hypothesis_status(condition: bool, partial_condition: bool) -> str:
    if condition:
        return "validée"
    if partial_condition:
        return "partiellement validée"
    return "non validée"


def render_hypotheses_panel(dataframe: pd.DataFrame) -> None:
    st.markdown("### Hypothèses expérimentales")
    frame = dataframe.copy()
    by_code = frame.set_index("configuration") if "configuration" in frame else frame

    def value(code: str, column: str) -> float | None:
        if code not in by_code.index or column not in by_code.columns:
            return None
        raw = by_code.loc[code, column]
        if isinstance(raw, pd.Series):
            raw = raw.iloc[0]
        return None if pd.isna(raw) else float(raw)

    c1_final = value("C1", "final_score")
    c3_final = value("C3", "final_score")
    c4_final = value("C4", "final_score")
    c1_rule = value("C1", "rule_compliance_score") or value("C1", "conops_rule_score")
    c3_rule = value("C3", "rule_compliance_score") or value("C3", "conops_rule_score")
    c1_grounding = value("C1", "grounding_score")
    c4_grounding = value("C4", "grounding_score")
    c5_grounding = value("C5", "grounding_score")
    c4_quality = c4_final
    c5_quality = value("C5", "final_score")
    c3_quality = c3_final
    c4_review = value("C4", "human_review_rate")
    c5_review = value("C5", "human_review_rate")
    h1_valid = (
        c1_final is not None
        and c3_final is not None
        and c3_final > c1_final
    ) or (
        c1_rule is not None
        and c3_rule is not None
        and c3_rule > c1_rule
    )
    h2_final_gain = (
        c1_final is not None and c4_final is not None and c4_final > c1_final
    )
    h2_grounding_gain = (
        c1_grounding is not None
        and c4_grounding is not None
        and c4_grounding > c1_grounding
    )
    h3_review_stable = (
        c4_review is None
        or c5_review is None
        or c5_review <= c4_review + 0.10
    )
    h3_final_gain = (
        c4_quality is not None and c5_quality is not None and c5_quality > c4_quality
    )
    h3_grounding_gain = (
        c4_grounding is not None
        and c5_grounding is not None
        and c5_grounding > c4_grounding
    )
    h3_below_c4_c3 = (
        c5_quality is not None
        and c4_quality is not None
        and c3_quality is not None
        and c5_quality < c4_quality
        and c5_quality < c3_quality
    )
    if h3_final_gain and h3_review_stable:
        h3_status = "validée"
    elif h3_grounding_gain and not h3_final_gain:
        h3_status = "partiellement validée"
    elif h3_below_c4_c3:
        h3_status = "non validée"
    else:
        h3_status = "partiellement validée" if c5_quality is not None else "non validée"

    rows = [
        {
            "Hypothèse": "H1 : les règles officielles améliorent la conformité ConOps/OpsCon.",
            "Statut": "validée" if h1_valid else "non validée",
            "Preuve observée": (
                f"final C1={c1_final}, C3={c3_final}; "
                f"règles C1={c1_rule}, C3={c3_rule}"
            ),
            "Limite": "La conformité reste automatique et demande une validation experte.",
        },
        {
            "Hypothèse": "H2 : le RAG améliore le grounding et la traçabilité.",
            "Statut": (
                "validée"
                if h2_final_gain and h2_grounding_gain
                else "partiellement validée"
                if h2_grounding_gain
                else "non validée"
            ),
            "Preuve observée": (
                f"final C1={c1_final}, C4={c4_final}; "
                f"grounding C1={c1_grounding}, C4={c4_grounding}"
            ),
            "Limite": "La qualité dépend du corpus indexé, du top-k et des chunks réellement utilisés.",
        },
        {
            "Hypothèse": "H3 : l’évaluateur améliore la qualité mais peut augmenter la revue humaine.",
            "Statut": h3_status,
            "Preuve observée": (
                f"score C3={c3_quality}, C4={c4_quality}, C5={c5_quality}; "
                f"grounding C4={c4_grounding}, C5={c5_grounding}; "
                f"revue C4={c4_review}, C5={c5_review}"
            ),
            "Limite": "Un meilleur score peut augmenter les éléments signalés pour revue humaine.",
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def build_hypotheses_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    ablation = build_ablation_analysis(dataframe)
    by_comparison = {
        row["comparison"]: row for row in ablation.to_dict(orient="records")
    }
    h1_gain = safe_float(by_comparison.get("C3 - C1", {}).get("score_delta"))
    h2_score_gain = safe_float(by_comparison.get("C4 - C1", {}).get("score_delta"))
    h2_grounding_gain = safe_float(
        by_comparison.get("C4 - C1", {}).get("grounding_delta")
    )
    h3_score_gain = safe_float(by_comparison.get("C5 - C4", {}).get("score_delta"))
    h3_review_delta = safe_float(
        by_comparison.get("C5 - C4", {}).get("human_review_delta")
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "statement": "Les règles officielles améliorent la conformité.",
                "status": "validée" if h1_gain >= 2 else "gain faible" if h1_gain >= 1 else "à confirmer",
                "evidence": f"C3-C1 score_delta={h1_gain}",
            },
            {
                "hypothesis": "H2",
                "statement": "Le RAG améliore grounding et score final.",
                "status": "validée" if h2_score_gain >= 2 and h2_grounding_gain > 0 else "partiellement validée" if h2_grounding_gain > 0 else "à confirmer",
                "evidence": f"C4-C1 score_delta={h2_score_gain}, grounding_delta={h2_grounding_gain}",
            },
            {
                "hypothesis": "H3",
                "statement": "L’évaluateur C5 améliore la qualité sans sur-revue.",
                "status": "validée" if h3_score_gain >= 2 and h3_review_delta <= 0.10 else "gain faible" if h3_score_gain >= 1 else "à confirmer",
                "evidence": f"C5-C4 score_delta={h3_score_gain}, human_review_delta={h3_review_delta}",
            },
        ]
    )


def _metric_delta(frame: pd.DataFrame, left: str, right: str, column: str) -> float | None:
    if "configuration" not in frame or column not in frame:
        return None
    indexed = frame.set_index("configuration")
    if left not in indexed.index or right not in indexed.index:
        return None
    left_value = pd.to_numeric(pd.Series([indexed.loc[left, column]]).iloc[0], errors="coerce")
    right_value = pd.to_numeric(pd.Series([indexed.loc[right, column]]).iloc[0], errors="coerce")
    if pd.isna(left_value) or pd.isna(right_value):
        return None
    return round(safe_float(left_value) - safe_float(right_value), 3)


def build_ablation_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("C3 - C1", "Effet des règles officielles", "C3", "C1"),
        ("C4 - C1", "Effet du RAG", "C4", "C1"),
        ("C5 - C4", "Effet de l’évaluateur LLM", "C5", "C4"),
        ("C5 - C3", "Effet RAG + évaluateur vs règles seules", "C5", "C3"),
    ]
    rows = []
    for comparison, component, left, right in comparisons:
        score_delta = _metric_delta(dataframe, left, right, "final_score")
        grounding_delta = _metric_delta(dataframe, left, right, "grounding_score")
        semantic_delta = _metric_delta(
            dataframe,
            left,
            right,
            "semantic_similarity",
        )
        review_delta = _metric_delta(dataframe, left, right, "human_review_rate")
        if score_delta is None:
            interpretation = "Donnée non disponible pour cette comparaison."
        elif score_delta > 0:
            interpretation = f"{component} améliore le score final."
        elif grounding_delta is not None and grounding_delta > 0:
            interpretation = f"{component} améliore le grounding mais pas le score final."
        else:
            interpretation = f"{component} n’apporte pas de gain mesurable ici."
        rows.append(
            {
                "comparison": comparison,
                "tested_component": component,
                "score_delta": score_delta,
                "grounding_delta": grounding_delta,
                "semantic_similarity_delta": semantic_delta,
                "human_review_delta": review_delta,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def automatic_scientific_conclusion(dataframe: pd.DataFrame) -> str:
    if dataframe.empty or "configuration" not in dataframe:
        return "Conclusion non disponible : aucun résultat exploitable."
    frame = dataframe.copy()
    for column in (
        "final_score",
        "grounding_score",
        "semantic_similarity",
        "human_review_rate",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    best_row = frame.sort_values("final_score", ascending=False).iloc[0]
    best_code = str(best_row["configuration"])
    indexed = frame.set_index("configuration")
    grounding_best = (
        indexed["grounding_score"].map(safe_float).idxmax()
        if "grounding_score" in indexed
        else "non disponible"
    )
    semantic_best = (
        indexed["semantic_similarity"].map(safe_float).idxmax()
        if "semantic_similarity" in indexed
        else "non disponible"
    )
    second_score = (
        safe_float(frame.sort_values("final_score", ascending=False).iloc[1]["final_score"])
        if len(frame) > 1
        else 0.0
    )
    best_gain = safe_float(best_row.get("final_score")) - second_score
    if best_gain >= 2.0:
        parts = [f"La meilleure configuration finale est {best_code} avec un gain net."]
    elif best_gain >= 1.0:
        parts = [
            f"La configuration {best_code} arrive en tête, mais le gain est faible "
            "et reste à confirmer."
        ]
    else:
        parts = [
            f"La configuration {best_code} arrive en tête, mais l’écart est non "
            "significatif ou à confirmer."
        ]
    if best_code == "C3":
        c1_grounding = (
            safe_float(indexed.loc["C1", "grounding_score"])
            if "C1" in indexed.index and "grounding_score" in indexed
            else None
        )
        rag_grounding = [
            code
            for code in ("C4", "C5")
            if code in indexed.index
            and c1_grounding is not None
            and safe_float(indexed.loc[code, "grounding_score"]) > c1_grounding
        ]
        parts.append(
            "Cette configuration n’utilise pas le RAG ni l’évaluateur LLM."
        )
        if rag_grounding:
            parts.append(
                "Cependant, C4 et C5 ont bien utilisé le RAG et ont amélioré "
                "le grounding documentaire."
            )
    parts.append(
        f"La configuration qui améliore le plus le grounding est {grounding_best}."
    )
    parts.append(
        f"La meilleure similarité sémantique est observée pour {semantic_best}."
    )
    ablation = build_ablation_analysis(dataframe)
    useful = {
        row["comparison"]: row
        for row in ablation.to_dict(orient="records")
    }
    delta_c3_c1 = safe_float(useful.get("C3 - C1", {}).get("score_delta"))
    delta_c4_c1 = safe_float(useful.get("C4 - C1", {}).get("score_delta"))
    grounding_delta_c4_c1 = safe_float(
        useful.get("C4 - C1", {}).get("grounding_delta")
    )
    delta_c5_c4 = safe_float(useful.get("C5 - C4", {}).get("score_delta"))
    if delta_c3_c1 >= 2.0:
        parts.append("Sur ce document, les règles officielles V3.0 apportent un gain fort.")
    elif delta_c3_c1 >= 1.0:
        parts.append("Les règles officielles V3.0 apportent un gain faible, à confirmer.")
    if (
        grounding_delta_c4_c1 > 0
        and delta_c4_c1 <= 0
    ):
        parts.append(
            "Le RAG améliore le grounding documentaire, mais ne suffit pas à "
            "améliorer le score final."
        )
    if delta_c5_c4 <= -2.0:
        parts.append(
            "L’évaluateur C5 demande un recalibrage car il diminue le score final "
            "par rapport à C4."
        )
    elif delta_c5_c4 < 0:
        parts.append(
            "L’évaluateur C5 est légèrement inférieur à C4; l’écart est faible "
            "et reste à confirmer."
        )
    return " ".join(parts)


def build_stability_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in dataframe.to_dict(orient="records"):
        mean_score = safe_float(row.get("mean_score") or row.get("final_score"))
        std_score = safe_float(row.get("std_score"))
        rows.append(
            {
                "configuration": row.get("configuration", ""),
                "repeat_count": row.get("repeat_count", 1),
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": row.get("min_score", mean_score),
                "max_score": row.get("max_score", mean_score),
                "stability_score": row.get("stability_score", max(0.0, 100.0 - std_score)),
                "coefficient_of_variation": round(std_score / mean_score, 4)
                if mean_score
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_configuration_kpi_table(row: pd.Series | dict) -> pd.DataFrame:
    """Build a readable, configuration-specific experimental KPI table."""
    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    def number(field: str, decimals: int = 3) -> float:
        return round(safe_float(values.get(field)), decimals)

    mean_score = safe_float(values.get("mean_score") or values.get("final_score"))
    std_score = safe_float(values.get("std_score"))
    stability_score = safe_float(
        values.get("stability_score", max(0.0, 100.0 - std_score))
    )
    coefficient = std_score / mean_score if mean_score else 0.0
    json_rate = safe_float(
        values.get(
            "json_validity_rate",
            1.0 if bool(values.get("json_valid", False)) else 0.0,
        )
    )
    rows = [
        ("Performance", "Score final moyen", round(mean_score, 3), "/ 100"),
        ("Performance", "Score minimal", number("min_score") or round(mean_score, 3), "/ 100"),
        ("Performance", "Score maximal", number("max_score") or round(mean_score, 3), "/ 100"),
        ("Stabilité", "Écart-type du score", round(std_score, 3), "plus faible = mieux"),
        ("Stabilité", "Coefficient de variation", round(coefficient, 4), "plus faible = mieux"),
        ("Stabilité", "Indice de stabilité", round(stability_score, 3), "/ 100"),
        ("Fiabilité JSON", "Taux de JSON valide", round(json_rate * 100.0, 1), "%"),
        ("Règles", "Conformité ConOps", number("conops_rule_score"), "/ 100"),
        ("Règles", "Conformité OpsCon", number("opscon_rule_score"), "/ 100"),
        ("Règles", "Pénalité règles interdites", number("forbidden_penalty_total"), "plus faible = mieux"),
        ("Traçabilité", "Grounding documentaire", number("grounding_score"), "/ 1"),
        ("Traçabilité", "Traçabilité RAG", number("rag_traceability_score"), "/ 1"),
        ("Qualité", "Similarité sémantique", number("semantic_similarity"), "/ 1"),
        ("Qualité", "Couverture des concepts", number("concept_coverage_score"), "/ 1"),
        ("Modèle", "Validation SysML", number("sysml_score"), "/ 100"),
        ("Extraction", "Exigences extraites", int(number("requirements_count", 0)), "éléments"),
        ("Extraction", "Stakeholders extraits", int(number("stakeholders_count", 0)), "éléments"),
        ("Extraction", "Interfaces extraites", int(number("interfaces_count", 0)), "éléments"),
        ("Extraction", "Risques extraits", int(number("risks_count", 0)), "éléments"),
        ("Coût", "Temps moyen d'exécution", number("execution_time_sec"), "secondes"),
        ("Protocole", "Nombre de répétitions", int(number("repeat_count", 0) or 1), "runs"),
    ]
    return pd.DataFrame(rows, columns=["Famille", "KPI / métrique", "Valeur", "Unité / lecture"])


def build_configuration_delta_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Expose the concrete marginal contribution of rules, RAG and evaluator."""
    if dataframe.empty or "configuration" not in dataframe:
        return pd.DataFrame()
    indexed = dataframe.drop_duplicates("configuration").set_index("configuration")
    comparisons = [
        ("C3 − C1", "Apport des règles officielles", "C3", "C1"),
        ("C4 − C3", "Apport du RAG", "C4", "C3"),
        ("C5 − C4", "Apport de l'évaluateur LLM", "C5", "C4"),
    ]
    rows = []
    for comparison, contribution, after, before in comparisons:
        if after not in indexed.index or before not in indexed.index:
            continue
        rows.append(
            {
                "Comparaison": comparison,
                "Composant évalué": contribution,
                "Δ score final": round(
                    safe_float(indexed.loc[after].get("final_score"))
                    - safe_float(indexed.loc[before].get("final_score")),
                    3,
                ),
                "Δ grounding": round(
                    safe_float(indexed.loc[after].get("grounding_score"))
                    - safe_float(indexed.loc[before].get("grounding_score")),
                    3,
                ),
                "Δ stabilité": round(
                    safe_float(indexed.loc[after].get("stability_score"))
                    - safe_float(indexed.loc[before].get("stability_score")),
                    3,
                ),
                "Δ temps (s)": round(
                    safe_float(indexed.loc[after].get("execution_time_sec"))
                    - safe_float(indexed.loc[before].get("execution_time_sec")),
                    3,
                ),
            }
        )
    return pd.DataFrame(rows)


def render_score_badges(dataframe: pd.DataFrame, repeat_count: int) -> None:
    badges = []
    std_score = safe_float(
        pd.to_numeric(dataframe.get("std_score"), errors="coerce")
        .fillna(0)
        .mean()
    )
    badges.append("Préliminaire" if repeat_count < 5 else "Robuste" if std_score <= 5 else "Robuste à confirmer")
    indexed = dataframe.set_index("configuration") if "configuration" in dataframe else dataframe
    if {"C1", "C4"}.issubset(indexed.index) and safe_float(indexed.loc["C4", "grounding_score"]) > safe_float(indexed.loc["C1", "grounding_score"]):
        badges.append("RAG utile pour grounding")
    if {"C4", "C5"}.issubset(indexed.index) and safe_float(indexed.loc["C5", "final_score"]) < safe_float(indexed.loc["C4", "final_score"]):
        badges.append("Evaluateur à recalibrer")
    st.write(" ".join(f"`{badge}`" for badge in badges))


def build_multi_document_summary() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(OUTPUTS_DIR.glob("run_*/experiments_summary.csv")):
        try:
            frame = pd.read_csv(summary_path)
        except Exception:
            continue
        if frame.empty or "configuration" not in frame:
            continue
        numeric_columns = [
            "final_score",
            "grounding_score",
            "semantic_similarity",
            "human_review_rate",
            "stability_score",
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).fillna(0)
        successful = frame[frame.get("status", "success") == "success"].copy()
        if successful.empty:
            successful = frame.copy()
        best = successful.sort_values("final_score", ascending=False).iloc[0]
        indexed = successful.set_index("configuration")
        def score(code: str) -> float | None:
            return (
                safe_float(indexed.loc[code, "final_score"])
                if code in indexed.index and "final_score" in indexed
                else 0.0
            )
        grounding_best = (
            indexed["grounding_score"].map(safe_float).idxmax()
            if "grounding_score" in indexed and not indexed.empty
            else ""
        )
        rows.append(
            {
                "document_name": best.get("document_name", ""),
                "best_configuration": best.get("configuration", ""),
                "C1_score": score("C1"),
                "C3_score": score("C3"),
                "C4_score": score("C4"),
                "C5_score": score("C5"),
                "best_reason": automatic_scientific_conclusion(successful),
                "grounding_best": grounding_best,
                "stability_score": safe_float(best.get("stability_score")),
            }
        )
    return pd.DataFrame(rows)


def load_all_run_summaries() -> pd.DataFrame:
    frames = []
    for summary_path in sorted(OUTPUTS_DIR.glob("run_*/experiments_summary.csv")):
        try:
            frame = pd.read_csv(summary_path)
        except Exception:
            continue
        if frame.empty or "configuration" not in frame:
            continue
        frame["run_dir"] = str(summary_path.parent)
        for column in [
            "final_score",
            "grounding_score",
            "semantic_similarity",
            "human_review_rate",
            "json_validity_rate",
            "extraction_consistency_score",
            "rag_traceability_score",
            "rule_score_consistency",
            "std_score",
            "stability_score",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_multi_document_config_stats() -> pd.DataFrame:
    frame = load_all_run_summaries()
    if frame.empty or "configuration" not in frame:
        return pd.DataFrame()
    rows = []
    for configuration, group in frame.groupby("configuration"):
        scores = pd.to_numeric(group["final_score"], errors="coerce").dropna()
        n = len(scores)
        mean_score = safe_float(scores.mean())
        std_score = safe_float(scores.std(ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std_score / math.sqrt(n) if n > 1 else 0.0
        baseline_mean = mean_score
        if configuration != "C1":
            c1_scores = pd.to_numeric(
                frame[frame["configuration"] == "C1"]["final_score"],
                errors="coerce",
            ).dropna()
            baseline_mean = safe_float(c1_scores.mean())
        gain = mean_score - baseline_mean
        if configuration == "C1":
            conclusion = "Baseline."
        elif gain >= 2.0:
            conclusion = "gain fort"
        elif gain >= 1.0:
            conclusion = "gain faible"
        elif gain > -1.0:
            conclusion = "non significatif ou à confirmer"
        else:
            conclusion = "à confirmer"
        rows.append(
            {
                "configuration": configuration,
                "documents_count": n,
                "mean_score": round(mean_score, 3),
                "std_score": round(std_score, 3),
                "ci95_low": round(mean_score - ci95, 3),
                "ci95_high": round(mean_score + ci95, 3),
                "mean_grounding": round(safe_float(group.get("grounding_score", pd.Series()).mean()), 3),
                "json_validity": round(safe_float(group.get("json_validity_rate", pd.Series()).mean()), 3),
                "extraction_stability": round(safe_float(group.get("extraction_consistency_score", pd.Series()).mean()), 3),
                "rag_stability": round(safe_float(group.get("rag_traceability_score", pd.Series()).mean()), 3),
                "rule_consistency": round(safe_float(group.get("rule_score_consistency", pd.Series()).mean()), 3),
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


def automatic_experimental_limits(dataframe: pd.DataFrame, n_repeats: int) -> list[str]:
    limits = [
        "Les scores restent issus d’une évaluation automatique et ne remplacent pas une validation experte.",
        "La validation SysML est une validation heuristique non officielle.",
    ]
    if n_repeats < 5:
        limits.append(
            "Le nombre de répétitions est inférieur à 5; les conclusions sont préliminaires."
        )
    if len(dataframe.get("document_name", pd.Series(dtype=str)).dropna().unique()) <= 1:
        limits.append(
            "Un seul document est analysé dans ce run; la généralisation multi-documents est à confirmer."
        )
    if "rag_used" in dataframe and not dataframe["rag_used"].any():
        limits.append("Aucun RAG effectif n’a été observé pour ce run.")
    return limits


def build_typed_traceability_edges(
    detailed_dataframe: pd.DataFrame,
    rules_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for index, row in detailed_dataframe.iterrows():
        item_id = str(row.get("item_id", f"ITEM-{index}"))
        evidence = str(row.get("evidence_quote") or row.get("evidence_text") or "")
        rule_id = str(row.get("related_rule_id") or "")
        if evidence:
            rows.append(
                {
                    "source": item_id,
                    "target": f"EVID-{index}",
                    "relation_type": "supported_by",
                    "evidence": evidence,
                }
            )
            rows.append(
                {
                    "source": item_id,
                    "target": f"EVID-{index}",
                    "relation_type": "derived_from",
                    "evidence": evidence,
                }
            )
        if rule_id:
            rows.append(
                {
                    "source": item_id,
                    "target": rule_id,
                    "relation_type": "satisfies_rule",
                    "evidence": evidence,
                }
            )
        if str(row.get("item_type", "")) == "interface":
            source_actor = str(row.get("source_actor") or row.get("source") or "")
            target_actor = str(row.get("target_actor") or row.get("target") or "")
            if source_actor:
                rows.append(
                    {
                        "source": item_id,
                        "target": source_actor,
                        "relation_type": "connects_source",
                        "evidence": evidence,
                    }
                )
            if target_actor:
                rows.append(
                    {
                        "source": item_id,
                        "target": target_actor,
                        "relation_type": "connects_target",
                        "evidence": evidence,
                    }
                )
        rows.append(
            {
                "source": item_id,
                "target": "SYSML_EXPORT",
                "relation_type": "generated_from",
                "evidence": evidence,
            }
        )
    if not rules_dataframe.empty:
        for _, row in rules_dataframe.iterrows():
            rows.append(
                {
                    "source": str(row.get("configuration", "")),
                    "target": str(row.get("rule_id", "")),
                    "relation_type": "satisfies_rule",
                    "evidence": str(row.get("evidence_quote", "")),
                }
            )
    return pd.DataFrame(rows)


def write_scientific_report(
    path: Path,
    *,
    document_name: str,
    protocol_rows: list[tuple[str, object]],
    hypotheses: pd.DataFrame,
    ablation: pd.DataFrame,
    results: pd.DataFrame,
    limits: list[str],
    conclusion: str,
) -> Path:
    def table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except Exception:
            return frame.to_string(index=False)

    lines = [
        "# Rapport scientifique expérimental",
        "",
        "## Protocole",
        "",
        table(pd.DataFrame(protocol_rows, columns=["Paramètre", "Valeur"])),
        "",
        "## Hypothèses",
        "",
        table(hypotheses),
        "",
        "## Résultats",
        "",
        table(results),
        "",
        "## Analyse d’ablation",
        "",
        table(ablation),
        "",
        "## Limites expérimentales",
        "",
        *[f"- {limit}" for limit in limits],
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"Document analysé : {document_name}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_multi_document_summary() -> Path | None:
    frame = build_multi_document_summary()
    if frame.empty:
        return None
    global_dir = Path("global_results")
    global_dir.mkdir(parents=True, exist_ok=True)
    path = global_dir / "multi_document_summary.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    config_stats = build_multi_document_config_stats()
    if not config_stats.empty:
        config_stats.to_csv(
            global_dir / "multi_document_config_stats.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return path


def render_multi_document_summary_panel() -> None:
    frame = build_multi_document_summary()
    if frame.empty:
        st.info("Aucune synthèse multi-documents disponible.")
        return
    st.markdown("### Comparaison par document")
    st.dataframe(frame, use_container_width=True)
    config_stats = build_multi_document_config_stats()
    if not config_stats.empty:
        st.markdown("### Statistiques par configuration")
        st.dataframe(config_stats, use_container_width=True)
    path = export_multi_document_summary()
    if path and path.exists():
        render_download_button(
            path=path,
            run_id="global",
            scope="multi_document_summary",
            index=0,
            label="Télécharger multi_document_summary.csv",
        )


init_run_session_state()


with st.sidebar:
    st.header("Zone 1 - Document")
    st.caption("PDF ConOps, texte brut ou futur draft genere.")
    st.header("Zone 2 - Connaissance metier")
    st.caption(
        "Regles ConOps/OpsCon, template, ontologie et documents RAG."
    )
    rules_repository = load_official_rules_repository()
    rules_data = rules_repository.data
    scoring_policy = rules_data["scoring_policy"]
    st.success("Referentiel de regles officielles V3.0 charge.")
    st.write(f"Version des regles : `{rules_repository.version}`")
    st.write(f"Nombre de regles : `{len(rules_repository.all())}`")
    st.write("Integrite des regles : `valide`")
    st.write(f"Alpha regle manquante : `{scoring_policy['alpha']}`")
    st.write(f"Beta violation interdite : `{scoring_policy['beta']}`")
    fine_tuning_status = fine_tuning_readiness()
    if fine_tuning_status.is_ready:
        st.success("C6/C7 eligibles : adaptateur LoRA et dataset expert disponibles.")
    else:
        st.warning("C6/C7 conditionnels : adaptateur LoRA non disponible.")
        st.caption(fine_tuning_status.message())
    enable_expert_validation = st.checkbox(
        "Activer la validation expert",
        value=True,
    )
    st.header("Zone 3 - Strategie experimentale")
    provider_options = ["ollama", "openai", "mock"]
    provider_index = (
        provider_options.index(settings.llm_provider)
        if settings.llm_provider in provider_options
        else 0
    )
    selected_provider = st.selectbox(
        "Fournisseur LLM",
        provider_options,
        index=provider_index,
        format_func=lambda value: {
            "ollama": "Ollama (local)",
            "openai": "OpenAI",
            "mock": "Heuristique locale",
        }[value],
    )
    if selected_provider == "mock":
        st.warning(
            "Analyse heuristique locale active. La comparaison avec un LLM externe se fait avec Ollama ou OpenAI."
        )
    if selected_provider == "ollama":
        if st.button("Actualisation des modeles Ollama", key="refresh_ollama_models"):
            detected_ollama_models.clear()
            st.rerun()

    st.write(f"Dossier references : `{REFERENCE_DOCS_DIR}`")
    use_uploaded_refs = st.checkbox(
        "PDF de reference de cette session",
        value=True,
    )
    experiment_mode = st.selectbox(
        "Mode experimental",
        [
            "Comparaison des strategies",
            "Comparaison des modeles",
            "Comparaison des formats de connaissance",
        ],
    )
    fixed_model = st.selectbox(
        "Modele fixe",
        AVAILABLE_MODELS,
        index=1,
        disabled=experiment_mode == "Comparaison des modeles",
    )
    fixed_strategy = st.selectbox(
        "Strategie fixe",
        ["C1", "C3", "C4", "C5", "C6", "C7"],
        index=3,
        disabled=experiment_mode == "Comparaison des strategies",
    )
    knowledge_format = st.selectbox(
        "Format de connaissance",
        ["K1", "K2", "K3"],
        format_func=lambda value: {
            "K1": "K1 - liste simple",
            "K2": "K2 - concepts et definitions",
            "K3": "K3 - ontologie et relations",
        }[value],
        disabled=experiment_mode == "Comparaison des formats de connaissance",
    )
    evaluator_strictness = st.selectbox(
        "Strictness evaluateur C5",
        ["medium", "low", "high"],
        index=0,
        help=(
            "low limite les penalites C5, medium garde un equilibre, "
            "high correspond a une revue plus stricte."
        ),
    )
    selected_codes = (
        ["C1", "C3", "C4", "C5"]
        if experiment_mode == "Comparaison des strategies"
        else [fixed_strategy]
    )
    n_repeats = st.slider(
        "Repetitions par configuration",
        min_value=1,
        max_value=10,
        value=1,
        help="Trois repetitions donnent une comparaison experimentale plus stable.",
    )

    models_by_configuration: dict[str, str] = {}
    if selected_provider == "ollama":
        installed_models = detected_ollama_models()
        model_options = list(
            dict.fromkeys(
                installed_models
                + list(DEFAULT_MODELS.values())
                + ["mistral", "qwen2.5:3b", "llama3.2"]
            )
        )
        if installed_models:
            st.success(
                f"Ollama disponible : {len(installed_models)} modele(s) installe(s)."
            )
        else:
            st.error(
                "Ollama est inaccessible ou aucun modele n'est installe. "
                "Les noms configures restent disponibles."
            )
        if experiment_mode == "Comparaison des strategies":
            models_by_configuration = {
                code: fixed_model for code in selected_codes
            }
        elif experiment_mode == "Comparaison des formats de connaissance":
            models_by_configuration = {fixed_strategy: fixed_model}
        else:
            models_by_configuration = {fixed_strategy: AVAILABLE_MODELS[0]}
    elif selected_provider == "openai":
        openai_model = st.text_input("Modele OpenAI", value=settings.openai_model)
        fixed_model = openai_model
        models_by_configuration = {code: openai_model for code in selected_codes}


st.markdown("## 1. Document ConOps/OpsCon a analyser")
target_pdf = st.file_uploader("PDF cible utilisateur", type=["pdf"])
if target_pdf is not None:
    st.caption(f"Document cible : `{target_pdf.name}`")
expert_validation_file = st.file_uploader(
    "Validation expert optionnelle (JSON, facultatif)",
    type=["json"],
)
st.markdown("## 2. Documents de reference RAG")
ref_pdfs = st.file_uploader(
    "PDF de reference : Airbus, GOES-R, EUROCONTROL, U-space, SESAR...",
    type=["pdf"],
    accept_multiple_files=True,
)

if ref_pdfs and use_uploaded_refs:
    REFERENCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded_file in ref_pdfs:
        (REFERENCE_DOCS_DIR / uploaded_file.name).write_bytes(uploaded_file.getbuffer())
    st.success(
        f"{len(ref_pdfs)} PDF de reference copie(s) dans data/reference_docs."
    )

if st.button("Construction / reconstruction de l'index RAG"):
    if not list(REFERENCE_DOCS_DIR.glob("*.pdf")):
        st.warning(
            "Aucun PDF de reference disponible. L'index Chroma fonctionne "
            "avec au moins un document de reference."
        )
    else:
        try:
            with st.spinner(
                "Construction de l'index Chroma local et calcul "
                "des embeddings..."
            ):
                rag = RagEngine(REFERENCE_DOCS_DIR)
                rag.build()
            st.session_state["rag"] = rag
            st.success(
                f"Chroma index construit : {rag.nb_chunks} chunks."
            )
        except Exception as exc:
            st.error(f"Echec de construction de l'index Chroma : {exc}")

if "rag" not in st.session_state:
    st.session_state["rag"] = RagEngine(REFERENCE_DOCS_DIR)
current_rag = st.session_state["rag"]
st.caption(
    f"Backend RAG : `{current_rag.backend}` | "
    f"chunks indexes : `{current_rag.nb_chunks}`"
)


st.markdown("## 3. Analyse experimentale")
with st.expander("Protocole experimental", expanded=False):
    st.markdown(
        """
- Meme document cible pour toutes les configurations.
- Meme modele LLM, meme temperature et meme nombre de repetitions.
- Meme format JSON; seule la strategie experimentale varie.
- Tracabilite par `run_id`, `document_hash`, `config_hash` et `prompt_hash`.
"""
    )
run = st.button(
    "Analyser le ConOps et comparer les configurations",
    type="primary",
    disabled=target_pdf is None or not selected_codes,
)
render_reload_last_run_option()

if run and target_pdf:
    rag_requested = (
        experiment_mode == "Comparaison des strategies"
        or fixed_strategy in {"C4", "C5", "C7"}
    )
    if rag_requested and not st.session_state["rag"].is_ready:
        st.warning(
            "C4/C5/C7 demandent un index RAG. Aucun chunk Chroma "
            "n'est disponible pour cette analyse."
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
        temporary_file.write(target_pdf.getbuffer())
        temporary_path = Path(temporary_file.name)

    extraction_state = {"ocr_started": False}
    extraction_progress = st.progress(
        0.0,
        text="Extraction du texte PDF...",
    )

    def update_extraction_progress(
        completed: int,
        total: int,
        message: str,
    ) -> None:
        extraction_state["ocr_started"] = True
        ratio = completed / total if total else 0.0
        extraction_progress.progress(
            min(1.0, max(0.0, ratio)),
            text=message,
        )

    try:
        try:
            document_text = extract_pdf_text(
                temporary_path,
                progress_callback=update_extraction_progress,
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    finally:
        temporary_path.unlink(missing_ok=True)
    extraction_progress.progress(1.0, text="Extraction PDF terminee.")
    if extraction_state["ocr_started"]:
        st.info(
            "PDF scanne detecte : OCR local utilise. Le resultat est mis "
            "en cache pour accelerer les prochaines analyses."
        )

    if len(document_text.strip()) < 200:
        st.error(
            "Le PDF contient trop peu de texte extractible. "
            "Un OCR est probablement necessaire."
        )
        st.stop()

    with st.spinner("Construction du texte de reference..."):
        reference_pages = load_pdfs_from_folder(REFERENCE_DOCS_DIR)
        reference_text = "\n".join(page.text for page in reference_pages[:80])

    with st.spinner("Appel LLM / RAG / evaluateur..."):
        rag = st.session_state.get("rag") or RagEngine(REFERENCE_DOCS_DIR)
        if expert_validation_file is not None:
            st.info(
                "Le JSON charge est traite comme support de validation expert, "
                "pas comme reference de verite document par document."
            )
        runner = ExperimentRunner(
            rag=rag,
            reference_text=reference_text,
            provider=selected_provider,
            models_by_configuration=models_by_configuration,
            knowledge_format=knowledge_format,
            evaluator_strictness=evaluator_strictness,
        )
        if experiment_mode == "Comparaison des strategies":
            results, analyses, reports = runner.run_strategy_comparison(
                document_text,
                fixed_model=fixed_model,
                document_name=target_pdf.name,
                n_repeats=n_repeats,
            )
        elif experiment_mode == "Comparaison des modeles":
            results, analyses, reports = runner.run_model_comparison(
                document_text,
                fixed_strategy=fixed_strategy,
                models=AVAILABLE_MODELS,
                document_name=target_pdf.name,
                n_repeats=n_repeats,
            )
        else:
            results, analyses, reports = (
                runner.run_knowledge_format_comparison(
                    document_text,
                    fixed_model=fixed_model,
                    fixed_strategy=fixed_strategy,
                    document_name=target_pdf.name,
                    n_repeats=n_repeats,
                )
            )
        dataframe = results_to_dataframe(results)
        dataframe = enrich_experimental_dataframe(dataframe)

    for code, error in runner.errors.items():
        st.warning(f"Configuration {code} : {error}")

    successful_dataframe = dataframe[dataframe["status"] == "success"].copy()
    if successful_dataframe.empty:
        st.markdown("## Resultats comparatifs")
        st.dataframe(dataframe, use_container_width=True)
        st.error(
            "Aucune configuration n'a abouti. Le fournisseur, les modeles "
            "et le delai OLLAMA_TIMEOUT_SECONDS sont a controler."
        )
        st.stop()

    st.success("Analyse terminee.")
    st.code(f"run_id = {runner.run_id}", language="text")
    st.caption(
        "Resultat evalue automatiquement selon les regles V3.0, "
        "non valide par un expert metier."
    )
    st.caption("Le concept ambigu generique est volontairement exclu.")
    st.caption(
        "Le System-of-Systems est evalue uniquement au niveau OpsCon."
    )
    with st.expander("Formule du score final"):
        st.markdown(
            """
            `Score final = 70% evaluation deterministe des regles V3.0
            + 30% controles techniques de qualite, grounding et SysML.`

            Les violations interdites utilisent une penalite beta superieure
            a la penalite alpha appliquee aux elements attendus manquants.
            """
        )
    st.markdown("## Resultats comparatifs")
    st.dataframe(dataframe, use_container_width=True)

    color_map = {
        "C1": "#636EFA",
        "C2": "#EF553B",
        "C3": "#00CC96",
        "C4": "#AB63FA",
        "C5": "#FFA15A",
        "C6": "#19D3F3",
        "C7": "#FF6692",
    }
    chart_col1, chart_col2 = st.columns(2)
    score_figure = px.bar(
        dataframe,
        x="configuration",
        y="final_score",
        color="configuration",
        color_discrete_map=color_map,
        text_auto=".2f",
        title=f"Score final - {target_pdf.name}",
        hover_data=["status", "error_message", "model_name"],
    )
    score_figure.update_yaxes(range=[0, 100])
    chart_col1.plotly_chart(score_figure, use_container_width=True)

    similarity_figure = px.bar(
        dataframe,
        x="configuration",
        y="semantic_similarity",
        color="configuration",
        color_discrete_map=color_map,
        text_auto=".3f",
        title="Similarite semantique",
        hover_data=["status", "model_name"],
    )
    similarity_figure.update_yaxes(range=[0, 1])
    chart_col2.plotly_chart(similarity_figure, use_container_width=True)

    component_scores = successful_dataframe.melt(
        id_vars=["configuration"],
        value_vars=["conops_score", "sysml_score", "grounding_score"],
        var_name="critere",
        value_name="score",
    )
    component_scores.loc[
        component_scores["critere"] == "grounding_score",
        "score",
    ] *= 100
    component_figure = px.bar(
        component_scores,
        x="configuration",
        y="score",
        color="critere",
        barmode="group",
        text_auto=".1f",
        title="Scores ConOps, SysML et grounding",
    )
    component_figure.update_yaxes(range=[0, 100])
    st.plotly_chart(component_figure, use_container_width=True)

    criteria_dataframe = evaluation_criteria_dataframe(results, reports)
    if not criteria_dataframe.empty:
        heatmap_data = criteria_dataframe.pivot(
            index="criterion_name",
            columns="configuration",
            values="score",
        )
        heatmap_figure = px.imshow(
            heatmap_data,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            zmin=0,
            zmax=100,
            title="Heatmap configurations x criteres d'evaluation",
        )
        st.plotly_chart(heatmap_figure, use_container_width=True)

    count_columns = [
        "requirements_count",
        "stakeholders_count",
        "risks_count",
        "interfaces_count",
        "future_actions_count",
    ]
    counts_dataframe = successful_dataframe.melt(
        id_vars=["configuration"],
        value_vars=count_columns,
        var_name="element",
        value_name="nombre",
    )
    counts_figure = px.bar(
        counts_dataframe,
        x="configuration",
        y="nombre",
        color="element",
        barmode="group",
        text_auto=True,
        title="Nombre d'elements extraits",
    )
    st.plotly_chart(counts_figure, use_container_width=True)

    best_code = successful_dataframe.sort_values(
        "final_score",
        ascending=False,
    ).iloc[0][
        "configuration"
    ]
    best_analysis = analyses[best_code]
    best_report = reports[best_code]
    best_strategy_code = successful_dataframe.sort_values(
        "final_score",
        ascending=False,
    ).iloc[0]["strategy_name"]
    best_configuration = next(
        configuration
        for configuration in CONFIGURATIONS
        if configuration.code == best_strategy_code
    )
    st.info(
        f"Meilleure configuration : **{best_code}** "
        f"avec {best_report.final_score}/100."
    )
    st.metric(
        "Taux de revue humaine",
        f"{best_report.human_review_rate:.1%}",
        help="Part des Ã©lÃ©ments traÃ§ables nÃ©cessitant une validation ou correction.",
    )
    best_explanation = explain_best_configuration(dataframe, best_code)
    st.markdown("### Pourquoi cette configuration gagne ?")
    for reason in best_explanation["bullets"]:
        st.write("-", reason)
    st.markdown("### Conclusion ConOps/OpsCon")
    rule_explanation = explain_classification(
        {
            "classification_by_rules": best_report.rule_based_classification,
            "conops_rule_score": best_report.conops_rule_score,
            "opscon_rule_score": best_report.opscon_rule_score,
            "rule_assessment_detailed": best_report.rule_assessment_detailed,
        },
        document_title=target_pdf.name,
    )
    st.write(
        "Le document presente une dominante "
        f"**{best_report.rule_based_classification}** selon les regles, "
        "a valider par expert."
    )
    for explanation_line in rule_explanation["bullets"]:
        st.write("-", explanation_line)
    title_lower = target_pdf.name.lower()
    if (
        "opscon" in title_lower
        and best_report.rule_based_classification == "ConOps"
    ):
        st.warning(
            "Le titre suggere OpsCon, mais le contenu detecte est "
            "majoritairement ConOps selon les regles. "
            "Validation experte recommandee."
        )
    if (
        "conops" in title_lower
        and "opscon" not in title_lower
        and best_report.rule_based_classification == "OpsCon"
    ):
        st.warning(
            "Le titre suggere ConOps, mais le contenu detecte est "
            "majoritairement OpsCon selon les regles. "
            "Validation experte recommandee."
        )

    run_dir = OUTPUTS_DIR / f"run_{runner.run_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []
    for code, analysis in analyses.items():
        json_path = run_dir / f"analysis_{code}.json"
        report_path = run_dir / f"report_{code}.md"
        sysml_path = run_dir / f"model_{code}.sysml"
        save_json(json_path, analysis)
        save_sysml(sysml_path, analysis)
        write_markdown_report(
            report_path,
            analysis,
            reports[code],
            dataframe,
            run_id=runner.run_id,
        )
        generated_paths.extend([json_path, report_path, sysml_path])
    graph_path = run_dir / "graph_2d.html"
    graph3d_path = run_dir / "graph_3d.html"
    save_pyvis_graph(best_analysis, graph_path)
    save_3d_graph(best_analysis, graph3d_path)
    save_graph_data(
        best_analysis,
        run_dir / "graph_data.json",
        run_dir / "graph_metrics.json",
    )
    save_sysml(run_dir / "model.sysml", best_analysis)
    sysml_exports = save_sysml_graph(best_analysis, run_dir)
    section_frame = export_section_classification(
        document_text,
        run_dir / "section_classification.csv",
    )
    save_json(
        run_dir / "report.json",
        {
            "analysis": best_analysis.model_dump(),
            "evaluation": best_report.model_dump(),
        },
    )
    save_json(run_dir / "classification_explanation.json", rule_explanation)
    write_markdown_report(
        run_dir / "report.md",
        best_analysis,
        best_report,
        dataframe,
        run_id=runner.run_id,
    )
    csv_paths = export_detailed_csv(
        results,
        analyses,
        reports,
        run_dir,
        rag_rows=runner.rag_rows,
    )
    csv_paths["summary"] = write_enriched_summary(run_dir, dataframe)
    ablation_dataframe = build_ablation_analysis(dataframe)
    ablation_path = run_dir / "ablation_analysis.csv"
    ablation_dataframe.to_csv(
        ablation_path,
        index=False,
        encoding="utf-8-sig",
    )
    metadata_path = write_run_metadata(
        run_dir,
        results,
        list(analyses),
        rag_documents=[path.name for path in REFERENCE_DOCS_DIR.glob("*.pdf")],
    )
    global_report_path = write_global_run_report(
        run_dir / "report_global.md",
        dataframe,
        runner.run_id,
    )
    detailed_dataframe = extractions_detailed_dataframe(results, analyses)
    detailed_dataframe[
        detailed_dataframe["human_review_needed"] == True
    ].to_csv(run_dir / "human_review_items.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_report.rule_assessment_detailed).to_csv(
        run_dir / "rule_compliance_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(runner.rag_rows).to_csv(
        run_dir / "rag_rule_traceability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rules_dataframe_for_traceability = pd.DataFrame(
        best_report.rule_assessment_detailed
    )
    typed_traceability_edges = build_typed_traceability_edges(
        detailed_dataframe,
        rules_dataframe_for_traceability,
    )
    typed_traceability_path = run_dir / "typed_traceability_edges.csv"
    typed_traceability_edges.to_csv(
        typed_traceability_path,
        index=False,
        encoding="utf-8-sig",
    )
    stability_report = analyze_repetitions(dataframe.to_dict(orient="records"))
    save_json(run_dir / "stability_report.json", stability_report)
    stability_summary = build_stability_summary(dataframe)
    stability_summary_path = run_dir / "stability_summary.csv"
    stability_summary.to_csv(
        stability_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    contradiction_detected = (
        "opscon" in target_pdf.name.lower()
        and best_report.rule_based_classification == "ConOps"
    )
    scientific_summary_path = write_scientific_summary(
        run_dir / "scientific_summary.md",
        document_name=target_pdf.name,
        contradiction_detected=contradiction_detected,
        final_classification=best_report.rule_based_classification,
        results=dataframe,
        best_explanation=best_explanation,
        section_frame=section_frame,
        stability_report=stability_report,
        rag_rows_count=len(runner.rag_rows),
    )
    protocol_rows = build_protocol_rows(
        document_name=target_pdf.name,
        dataframe=dataframe,
        n_repeats=n_repeats,
        model_name=str(successful_dataframe.iloc[0].get("model_name", fixed_model)),
        rag=st.session_state["rag"],
        rules_version=best_report.rules_version,
    )
    hypotheses_table = build_hypotheses_table(dataframe)
    experimental_limits = automatic_experimental_limits(dataframe, n_repeats)
    scientific_report_path = write_scientific_report(
        run_dir / "scientific_report.md",
        document_name=target_pdf.name,
        protocol_rows=protocol_rows,
        hypotheses=hypotheses_table,
        ablation=ablation_dataframe,
        results=dataframe,
        limits=experimental_limits,
        conclusion=automatic_scientific_conclusion(dataframe),
    )

    correction_csv = detailed_dataframe[
        [
            "configuration",
            "item_type",
            "item_id",
            "item_text",
            "validation_status",
            "error_flags",
        ]
    ].copy()
    correction_csv["human_correction"] = ""
    correction_csv["human_validation"] = ""
    correction_export_path = run_dir / f"human_review_{runner.run_id[:8]}.csv"
    correction_csv.to_csv(
        correction_export_path,
        index=False,
        encoding="utf-8-sig",
    )
    multi_document_config_stats = build_multi_document_config_stats()
    if not multi_document_config_stats.empty:
        multi_document_config_stats.to_csv(
            run_dir / "multi_document_config_stats.csv",
            index=False,
            encoding="utf-8-sig",
        )
    archive_path = zip_run_directory(run_dir)
    multi_document_summary_path = export_multi_document_summary()
    persist_last_run(
        run_id=runner.run_id,
        output_dir=run_dir,
        results=dataframe,
        generated_files=collect_generated_files(run_dir),
    )

    (
        tab1, tab2, tab3, tab4, tab5, tab6,
        tab7, tab8, tab9, tab10, tab11, tab12, tab13,
        tab14, tab15, tab16, tab17,
    ) = st.tabs(
        [
            "Evaluation",
            "Stakeholders",
            "Exigences",
            "Interfaces",
            "Risques",
            "Actions futures",
            "RAG chunks utilises",
            "Analyse par sections",
            "Revue humaine",
            "Evaluation regles officielles V3.0",
            "SysML v2",
            "Fichiers generes",
            "Regles ConOps/OpsCon",
            "Graphe de connaissances",
            "Stabilite experimentale",
            "Synthèse multi-documents",
            "Architecture & UML",
        ]
    )

    with tab1:
        st.subheader("Rapport d'evaluation")
        render_score_badges(dataframe, n_repeats)
        render_experimental_protocol_panel(
            document_name=target_pdf.name,
            dataframe=dataframe,
            n_repeats=n_repeats,
            model_name=best_report.model_name
            if hasattr(best_report, "model_name")
            else str(successful_dataframe.iloc[0].get("model_name", fixed_model)),
            rag=st.session_state["rag"],
            rules_version=best_report.rules_version,
        )
        render_hypotheses_panel(dataframe)
        st.markdown("### Analyse d’ablation")
        st.dataframe(ablation_dataframe, use_container_width=True)
        st.markdown("### Conclusion scientifique automatique")
        st.info(automatic_scientific_conclusion(dataframe))
        st.markdown("### Limites expérimentales")
        for limit in automatic_experimental_limits(dataframe, n_repeats):
            st.write("-", limit)
        metrics_dataframe = pd.DataFrame(
            [metric.model_dump() for metric in best_report.metrics]
        )
        st.dataframe(metrics_dataframe, use_container_width=True)
        quality_warnings = detailed_dataframe[
            detailed_dataframe["human_review_needed"] == True
        ]
        if not quality_warnings.empty:
            st.warning(
                f"{len(quality_warnings)} element(s) sont proposes pour revue humaine."
            )
            st.dataframe(
                quality_warnings[
                    [
                        "configuration",
                        "item_type",
                        "item_id",
                        "error_flags",
                        "item_quality_score",
                    ]
                ],
                use_container_width=True,
            )
        st.write("**Verdict :**", best_report.verdict)
        st.write(
            "**Revue humaine :**",
            f"{best_report.human_review_count} Ã©lÃ©ment(s), "
            f"soit {best_report.human_review_rate:.1%}.",
        )
        st.write("**Recommandations :**")
        for recommendation in best_report.recommendations:
            st.write("-", recommendation)
        st.markdown("### Gain par rapport au baseline C1")
        gain_dataframe = gain_vs_baseline(dataframe)
        st.dataframe(gain_dataframe, use_container_width=True)
        for interpretation in interpret_baseline_gain(dataframe):
            st.info(interpretation)
        st.markdown("### Pourquoi cette configuration gagne ?")
        for reason in best_explanation["bullets"]:
            st.write("-", reason)

    with tab2:
        st.subheader("Tableau des parties prenantes")
        stakeholders_dataframe = pd.DataFrame(
            [
                stakeholder.model_dump(exclude={"evidence"})
                for stakeholder in best_analysis.stakeholders
            ]
        )
        st.dataframe(stakeholders_dataframe, use_container_width=True)
        st.markdown("### Preuves")
        st.dataframe(
            detailed_dataframe[
                detailed_dataframe["item_type"] == "stakeholder"
            ][["item_id", "item_text", "evidence_quote", "source_page"]],
            use_container_width=True,
        )

    with tab3:
        st.subheader("Exigences haut niveau")
        st.dataframe(
            detailed_dataframe[
                detailed_dataframe["item_type"] == "requirement"
            ],
            use_container_width=True,
        )

    with tab4:
        st.subheader("Interfaces")
        st.dataframe(
            detailed_dataframe[
                detailed_dataframe["item_type"] == "interface"
            ],
            use_container_width=True,
        )

    with tab5:
        st.subheader("Risques")
        st.dataframe(
            detailed_dataframe[detailed_dataframe["item_type"] == "risk"],
            use_container_width=True,
        )

    with tab6:
        st.subheader("Actions, changements et points a eviter")
        st.markdown("### Actions futures")
        st.dataframe(
            pd.DataFrame(
                [action.model_dump() for action in best_analysis.future_actions]
            ),
            use_container_width=True,
        )
        st.markdown("### A changer")
        st.write(best_analysis.things_to_change)
        st.markdown("### A eviter")
        st.write(best_analysis.things_to_avoid)

    with tab7:
        st.subheader("Chunks RAG recuperes")
        rag_expected = any(
            result.status == "success"
            for result in results
            if result.configuration in {"C4", "C5", "C7"}
        )
        if rag_expected and not runner.rag_rows:
            st.error(
                "RAG activÃ© mais aucun chunk tracÃ© : rÃ©sultat non "
                "exploitable scientifiquement."
            )
        st.dataframe(
            dataframe_for_display(runner.rag_rows),
            use_container_width=True,
        )

    with tab8:
        st.subheader("Analyse par sections")
        st.caption(
            "Cette analyse explique les documents dont le titre suggere une "
            "famille mais dont certaines sections sont classees autrement."
        )
        st.dataframe(section_frame, use_container_width=True)

    with tab9:
        st.subheader("Validation expert")
        expert_frame = build_expert_validation_frame(
            detailed_dataframe,
            best_code,
        )
        if enable_expert_validation:
            edited_expert_frame = st.data_editor(
                expert_frame,
                use_container_width=True,
                disabled=[
                    column
                    for column in expert_frame.columns
                    if column not in {"expert_label", "expert_comment"}
                ],
                column_config={
                    "expert_label": st.column_config.SelectboxColumn(
                        options=[
                            "correct",
                            "partiellement_correct",
                            "faux",
                            "a_revoir",
                        ]
                    )
                },
            )
            save_expert_validation(
                edited_expert_frame,
                run_dir / "expert_validation.csv",
            )
        else:
            st.info("Validation expert desactivee pour ce run.")

    with tab10:
        st.subheader("Evaluation par regles officielles V3.0")
        st.info(
            "Les métriques supervisées reposent sur une annotation manuelle "
            "document par document. Cette plateforme utilise ici une évaluation "
            "déterministe fondée sur le référentiel de règles officielles V3.0, "
            "complétée par le grounding documentaire, la stabilité et la "
            "validation experte."
        )
        rule_cols = st.columns(4)
        rule_cols[0].metric(
            "Classification deterministe",
            best_report.rule_based_classification,
        )
        rule_cols[1].metric(
            "Score ConOps",
            f"{best_report.conops_rule_score:.1f}/100",
        )
        rule_cols[2].metric(
            "Score OpsCon",
            f"{best_report.opscon_rule_score:.1f}/100",
        )
        rule_cols[3].metric(
            "Penalite interdite",
            f"{best_report.forbidden_penalty:.1f}",
        )
        st.metric(
            "Regles obligatoires manquantes",
            best_report.missing_mandatory_rules_count,
        )
        st.markdown("### Matrice de conformite aux regles")
        st.dataframe(
            pd.DataFrame(best_report.rule_assessment_detailed),
            use_container_width=True,
        )
        st.caption(
            "Les resultats sont evalues automatiquement selon les regles "
            "officielles V3.0. Ils ne remplacent pas une validation experte metier."
        )

    with tab11:
        st.subheader("Fichier SysML v2 genere")
        sysml_text = generate_sysml_v2(best_analysis)
        sysml_score, sysml_errors = validate_sysml_text(sysml_text, best_analysis)
        st.write(
            f"Validation heuristique non officielle : "
            f"{sysml_score * 100:.1f}/100"
        )
        st.caption(
            "Cette validation heuristique non officielle ne remplace pas un parseur SysML v2 officiel."
        )
        if sysml_errors:
            st.warning(sysml_errors)
        st.code(sysml_text, language="java")

    with tab12:
        st.subheader("Telechargements")
        render_generated_files_panel()

    with tab13:
        st.subheader("Regles metier ConOps/OpsCon V3.0")
        st.write("**Version :**", best_report.rules_version)
        st.caption("Rules version: ConOps & OpsCon Evaluation Rules V3.0")
        st.write(
            "**Classification deterministe :**",
            best_report.rule_based_classification,
        )
        score_col1, score_col2 = st.columns(2)
        score_col1.metric(
            "Score regles ConOps",
            f"{best_report.conops_rule_score:.1f}/100",
        )
        score_col2.metric(
            "Score regles OpsCon",
            f"{best_report.opscon_rule_score:.1f}/100",
        )
        st.write(
            "**Penalite interdite totale :**",
            best_report.forbidden_penalty,
        )
        penalty_col1, penalty_col2 = st.columns(2)
        penalty_col1.metric(
            "Penalites ConOps",
            best_report.conops_forbidden_penalty,
        )
        penalty_col2.metric(
            "Penalites OpsCon",
            best_report.opscon_forbidden_penalty,
        )
        st.write(
            "**Regles obligatoires manquantes :**",
            best_report.missing_mandatory_rules_count,
        )
        rules_dataframe = dataframe_for_display(
            best_report.rule_assessment_detailed
        )
        if not rules_dataframe.empty:
            relevant_families = (
                ["D", best_report.rule_based_classification]
                if best_report.rule_based_classification
                in {"ConOps", "OpsCon"}
                else ["D", "ConOps", "OpsCon"]
            )
            st.markdown("### Regles ConOps")
            st.dataframe(
                rules_dataframe[
                    rules_dataframe["rule_family"].isin(["D", "ConOps"])
                ],
                use_container_width=True,
            )
            st.markdown("### Regles OpsCon")
            st.dataframe(
                rules_dataframe[
                    rules_dataframe["rule_family"].isin(["D", "OpsCon"])
                ],
                use_container_width=True,
            )
            violations = rules_dataframe[
                rules_dataframe["status"] == "forbidden_present"
            ]
            missing = rules_dataframe[
                (rules_dataframe["status"] == "missing")
                & (rules_dataframe["expected"] == "mandatory")
                & rules_dataframe["rule_family"].isin(relevant_families)
            ]
            st.markdown("### Violations interdites")
            st.dataframe(violations, use_container_width=True)
            st.markdown("### Regles obligatoires manquantes")
            st.dataframe(missing, use_container_width=True)
            st.markdown("### Revue humaine necessaire")
            st.dataframe(
                rules_dataframe[
                    (rules_dataframe["human_review_needed"] == True)
                    & rules_dataframe["rule_family"].isin(
                        relevant_families
                    )
                ],
                use_container_width=True,
            )

    with tab14:
        st.header("Graphes")
        import json as _json

        st.subheader("Graphe de connaissances métier")
        graph_metrics_path = run_dir / "graph_metrics.json"
        graph_data_path = run_dir / "graph_data.json"
        if graph_metrics_path.exists():
            st.json(_json.loads(graph_metrics_path.read_text(encoding="utf-8")))
        if graph_data_path.exists():
            graph_data = _json.loads(graph_data_path.read_text(encoding="utf-8"))
            node_types = sorted({node.get("node_type", "") for node in graph_data["nodes"]})
            selected_node_types = st.multiselect(
                "Filtre par type de noeud",
                node_types,
                default=node_types,
            )
            min_confidence = st.slider(
                "Confidence minimale",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
            )
            only_grounded = st.checkbox("Only grounded items", value=False)
            filtered_nodes = [
                node for node in graph_data["nodes"]
                if node.get("node_type", "") in selected_node_types
                and float(node.get("confidence", 0.0)) >= min_confidence
                and (
                    not only_grounded
                    or bool(node.get("evidence_quote") or node.get("source_page"))
                )
            ]
            st.dataframe(pd.DataFrame(filtered_nodes), use_container_width=True)
        if graph_path.exists():
            components.html(
                graph_path.read_text(encoding="utf-8"),
                height=760,
                scrolling=True,
            )
        st.subheader("Graphe de traçabilité RAG/règles")
        typed_traceability_path = run_dir / "typed_traceability_edges.csv"
        if typed_traceability_path.exists():
            st.caption(
                "Relations typées : supported_by, derived_from, satisfies_rule, "
                "generated_from, connects_source, connects_target."
            )
            st.dataframe(
                pd.read_csv(typed_traceability_path),
                use_container_width=True,
            )
        rag_traceability_path = run_dir / "rag_rule_traceability.csv"
        rule_edges_path = run_dir / "graph_edges.csv"
        if rag_traceability_path.exists():
            st.dataframe(pd.read_csv(rag_traceability_path), use_container_width=True)
        elif rule_edges_path.exists():
            st.dataframe(pd.read_csv(rule_edges_path), use_container_width=True)
        else:
            st.info("Aucune trace RAG/règles disponible pour ce run.")
        st.subheader("Graphe de traçabilité orienté SysML")
        st.caption(
            "Ce graphe est généré automatiquement à partir des entités extraites. "
            "Il ne constitue pas une validation officielle SysML v2."
        )
        sysml_graph_path = run_dir / "sysml_graph.html"
        if sysml_graph_path.exists():
            components.html(
                sysml_graph_path.read_text(encoding="utf-8"),
                height=760,
                scrolling=True,
            )

    with tab15:
        st.header("Stabilite experimentale")
        warning = repetition_warning(n_repeats)
        if warning:
            st.warning(warning)
        st.caption(
            "Chaque tableau présente les KPI propres à une configuration. "
            "Les configurations ne sont pas mélangées dans un JSON global."
        )
        configuration_descriptions = {
            "C1": "Référence — LLM seul",
            "C3": "LLM + règles ConOps/OpsCon",
            "C4": "LLM + règles + RAG",
            "C5": "LLM + règles + RAG + évaluateur",
        }
        indexed_results = (
            dataframe.drop_duplicates("configuration")
            .set_index("configuration")
            if "configuration" in dataframe
            else pd.DataFrame()
        )
        first_column, second_column = st.columns(2)
        for index, configuration in enumerate(("C1", "C3", "C4", "C5")):
            container = first_column if index % 2 == 0 else second_column
            with container:
                st.subheader(
                    f"{configuration} — "
                    f"{configuration_descriptions[configuration]}"
                )
                if configuration not in indexed_results.index:
                    st.info("Aucun résultat disponible pour cette configuration.")
                    continue
                st.dataframe(
                    build_configuration_kpi_table(
                        indexed_results.loc[configuration]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.subheader("Comparaison des apports entre configurations")
        delta_table = build_configuration_delta_table(dataframe)
        if delta_table.empty:
            st.info("Comparaison indisponible : résultats incomplets.")
        else:
            st.dataframe(
                delta_table,
                use_container_width=True,
                hide_index=True,
            )
        st.subheader("Conclusion fondée sur les KPI")
        st.info(automatic_scientific_conclusion(dataframe))

    with tab16:
        st.header("Synthèse multi-documents")
        render_multi_document_summary_panel()

    with tab17:
        st.header("Architecture UML de la plateforme")
        st.caption(
            "Le LLM extrait une structure JSON. Python applique ensuite "
            "les regles officielles V3.0 et calcule les scores."
        )
        for title, filename in UML_DIAGRAMS:
            diagram_path = UML_DIR / filename
            st.subheader(title)
            if diagram_path.exists():
                st.image(str(diagram_path), use_container_width=True)
            else:
                st.warning(f"Diagramme manquant : {diagram_path}")
elif st.session_state.get("analysis_done"):
    st.success("Dernier run disponible en session.")
    st.code(
        f"run_id = {st.session_state.get('last_run_id', '')}",
        language="text",
    )
    last_results = st.session_state.get("last_results")
    if last_results:
        st.markdown("## Resultats comparatifs")
        st.dataframe(dataframe_for_display(last_results), use_container_width=True)
    generated_tab, multi_doc_tab, uml_tab = st.tabs(
        ["Fichiers generes", "Synthèse multi-documents", "Architecture & UML"]
    )
    with generated_tab:
        st.subheader("Telechargements")
        render_generated_files_panel()
    with multi_doc_tab:
        st.header("Synthèse multi-documents")
        render_multi_document_summary_panel()
    with uml_tab:
        st.header("Architecture UML de la plateforme")
        for title, filename in UML_DIAGRAMS:
            diagram_path = UML_DIR / filename
            st.subheader(title)
            if diagram_path.exists():
                st.image(str(diagram_path), use_container_width=True)
            else:
                st.warning(f"Diagramme manquant : {diagram_path}")
else:
    st.info(
        "Les résultats apparaissent après une analyse."
    )
    with st.expander("Architecture & UML", expanded=False):
        st.caption(
            "Les diagrammes restent consultables avant le lancement "
            "d'une analyse."
        )
        for title, filename in UML_DIAGRAMS:
            diagram_path = UML_DIR / filename
            st.subheader(title)
            if diagram_path.exists():
                st.image(str(diagram_path), use_container_width=True)
            else:
                st.warning(f"Diagramme manquant : {diagram_path}")
