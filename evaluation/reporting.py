"""Reporters that serialize an :class:`EvalReport` to various formats.

Available reporters (registered via ``@register_reporter``):

* ``"console"`` -- human-readable stdout summary.
* ``"json"``    -- full report as ``eval_report.json``.
* ``"clearml"`` -- **Scalars** (overall KPIs for run comparison), bar-charts
  (PLOTS; one ClearML variant per headline metric for compare-friendly overlays),
  and summary tables sent to the active `ClearML <https://clear.ml>`_ task.
"""

from __future__ import annotations

import json
import logging
import numbers
from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Type

from evaluation.schema import EvalReport

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_REPORTER_REGISTRY: Dict[str, Type["Reporter"]] = {}


def register_reporter(name: str):
    """Class decorator that registers a reporter under *name*."""

    def decorator(cls: Type[Reporter]) -> Type[Reporter]:
        if name in _REPORTER_REGISTRY:
            raise ValueError(f"Reporter '{name}' already registered")
        _REPORTER_REGISTRY[name] = cls
        return cls

    return decorator


def build_reporter(formats: List[str]) -> Reporter:
    """Build a (possibly composite) reporter from a list of format names."""
    reporters: List[Reporter] = []
    for fmt in formats:
        if fmt not in _REPORTER_REGISTRY:
            raise KeyError(
                f"Unknown reporter format: '{fmt}'. "
                f"Available: {list(_REPORTER_REGISTRY)}"
            )
        reporters.append(_REPORTER_REGISTRY[fmt]())

    if len(reporters) == 1:
        return reporters[0]
    return CompositeReporter(reporters)


# ------------------------------------------------------------------
# Abstract interface
# ------------------------------------------------------------------

class Reporter(ABC):
    """Writes an :class:`EvalReport` to an output channel."""

    @abstractmethod
    def report(self, eval_report: EvalReport, output_dir: Path) -> None:
        ...


# ------------------------------------------------------------------
# Composite
# ------------------------------------------------------------------

class CompositeReporter(Reporter):
    """Delegates to multiple reporters."""

    def __init__(self, reporters: List[Reporter]):
        self._reporters = reporters

    def report(self, eval_report: EvalReport, output_dir: Path) -> None:
        for r in self._reporters:
            r.report(eval_report, output_dir)


# ------------------------------------------------------------------
# Concrete reporters
# ------------------------------------------------------------------

def _report_to_plain_dict(report: EvalReport) -> Dict[str, Any]:
    """Convert an EvalReport into a JSON-serializable dict."""
    sequences = {}
    for sid, sr in report.sequences.items():
        bins = {}
        for bname, br in sr.bins.items():
            bins[bname] = br.metric_values
        sequences[sid] = {
            "num_annotated_frames": sr.num_annotated_frames,
            "num_total_frames": sr.num_total_frames,
            "bins": bins,
        }

    aggregated = {
        bname: br.metric_values for bname, br in report.aggregated_bins.items()
    }

    return {
        "sequences": sequences,
        "aggregated": aggregated,
        "overall": report.overall,
    }


class _NanSafeEncoder(json.JSONEncoder):
    """Encode ``float('nan')``, ``inf``, and NumPy non-finite reals as ``null``.

    ``json.dump`` / ``iterencode`` do not call ``encode()``, so sanitization must
    run in ``iterencode`` for output to be strict JSON.
    """

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)

    def _sanitize(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, numbers.Real):
            xf = float(obj)
            if math.isnan(xf) or math.isinf(xf):
                return None
            return obj
        return obj


def _write_never_matched_txt(report: EvalReport, output_dir: Path) -> None:
    """Write ``never_matched.txt`` listing GT objects the tracker never detected.

    Format: ``video,frame,object_id`` (one row per frame appearance).
    Only written when at least one never-matched object exists.
    """
    rows: List[str] = []
    for sid, sr in report.sequences.items():
        for entry in sr.never_matched_gt:
            rows.append(f"{sid},{entry['frame_id']},{entry['object_id']}")
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "never_matched.txt"
    with open(out_path, "w") as fh:
        fh.write("video,frame,object_id\n")
        fh.write("\n".join(rows) + "\n")
    total_objects = len({
        (sid, e["object_id"])
        for sid, sr in report.sequences.items()
        for e in sr.never_matched_gt
    })
    logger.info(
        "Never-matched GT: %d unique objects, %d rows → %s",
        total_objects, len(rows), out_path,
    )


@register_reporter("json")
class JsonReporter(Reporter):
    """Write the full report as a JSON file."""

    def report(self, eval_report: EvalReport, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "eval_report.json"
        data = _report_to_plain_dict(eval_report)
        if eval_report.density_aggregated:
            data["density_aggregated"] = {
                dname: {
                    bname: br.metric_values
                    for bname, br in dbins.items()
                }
                for dname, dbins in eval_report.density_aggregated.items()
            }
        with open(out_path, "w") as fh:
            json.dump(data, fh, indent=2, cls=_NanSafeEncoder)
        logger.info("JSON report written to %s", out_path)
        _write_never_matched_txt(eval_report, output_dir)


def _group_slices_by_class(
    bins: Dict[str, Any],
) -> List[tuple[str, List[tuple[str, Any]]]]:
    """Group slice names by class prefix for hierarchical display.

    Slices with a ``" / "`` separator are grouped under their class prefix.
    Slices without a separator go into a ``""`` (top-level) group.

    Returns ``[(group_name, [(slice_name, bin_result), ...]), ...]``.
    """
    groups: Dict[str, List[tuple[str, Any]]] = {}
    for bname, br in bins.items():
        if " / " in bname:
            class_part, _size_part = bname.split(" / ", 1)
        else:
            class_part = ""
        groups.setdefault(class_part, []).append((bname, br))
    return list(groups.items())


@register_reporter("console")
class ConsoleReporter(Reporter):
    """Print a human-readable summary to stdout."""

    def report(self, eval_report: EvalReport, output_dir: Path) -> None:
        lines: List[str] = ["", "=" * 60, "  Evaluation Report", "=" * 60]

        for sid, sr in eval_report.sequences.items():
            lines.append(
                f"\nSequence: {sid}  "
                f"(annotated {sr.num_annotated_frames}/{sr.num_total_frames} frames)"
            )
            self._format_bins(sr.bins, lines)

        lines.append("\n" + "-" * 60)
        lines.append("  Aggregated (across sequences)")
        lines.append("-" * 60)
        self._format_bins(eval_report.aggregated_bins, lines)

        if eval_report.density_aggregated:
            lines.append("\n" + "-" * 60)
            lines.append("  Per Frame Density (GT objects/frame)")
            lines.append("-" * 60)
            for dname, dbins in eval_report.density_aggregated.items():
                overall_bin = dbins.get("all")
                if not overall_bin:
                    continue
                lines.append(f"  [{dname}]")
                for mname, values in overall_bin.metric_values.items():
                    parts = "  ".join(f"{k}={v:.4f}" for k, v in values.items())
                    lines.append(f"    {mname}: {parts}")

        total_nm = sum(
            len(sr.never_matched_gt) for sr in eval_report.sequences.values()
        )
        if total_nm > 0:
            unique_objs = len({
                (sid, e["object_id"])
                for sid, sr in eval_report.sequences.items()
                for e in sr.never_matched_gt
            })
            lines.append(f"\n  Never-matched GT: {unique_objs} objects "
                         f"({total_nm} frame appearances)")

        lines.append("")
        print("\n".join(lines))
        _write_never_matched_txt(eval_report, output_dir)

    @staticmethod
    def _format_bins(bins: Dict[str, Any], lines: List[str]) -> None:
        """Append formatted bin/slice lines with optional class grouping."""
        has_product_slices = any(" / " in name for name in bins)

        if not has_product_slices:
            for bname, br in bins.items():
                lines.append(f"  [{bname}]")
                for mname, values in br.metric_values.items():
                    parts = "  ".join(f"{k}={v:.4f}" for k, v in values.items())
                    lines.append(f"    {mname}: {parts}")
            return

        for group_name, items in _group_slices_by_class(bins):
            if group_name:
                lines.append(f"  -- {group_name} --")
            for bname, br in items:
                display = bname.split(" / ", 1)[1] if " / " in bname else bname
                indent = "    " if group_name else "  "
                lines.append(f"{indent}[{display}]")
                for mname, values in br.metric_values.items():
                    parts = "  ".join(f"{k}={v:.4f}" for k, v in values.items())
                    lines.append(f"{indent}  {mname}: {parts}")


# ------------------------------------------------------------------
# ClearML scalar KPIs (overall / aggregated "all" bin)
# ------------------------------------------------------------------

# Keys in ``EvalReport.overall`` metric groups to log as ClearML Scalars (not PLOTS).
# Excludes per-alpha HOTA breakdowns (``hota@*``) and per-cell PORR rates.
_OVERALL_SCALAR_KEYS: Dict[str, frozenset[str]] = {
    "hota": frozenset({"hota", "deta", "assa", "loca"}),
    "mota": frozenset({
        "mota", "motp", "recall", "precision", "far",
        "num_tp", "num_fp", "num_fn", "num_idsw", "frag",
        "mt", "pt", "ml", "num_gt", "num_gt_objects", "num_frames",
    }),
    "idf1": frozenset({"idf1", "idp", "idr", "idtp", "idfp", "idfn"}),
    "coverage": frozenset({
        "coverage", "coverage_per_track",
        "num_covered_frames", "num_visible_frames", "num_gt_tracks",
    }),
    "pd": frozenset({"pd", "num_detected", "num_gt_objects"}),
    "id_instability": frozenset({"id_instability", "num_gt_objects", "total_idsw"}),
    "realtime_kpi": frozenset({
        "tid_mean_frames", "tid_mean_sec", "tid_median_frames",
        "tid_ratio_immediate", "tid_ratio_within_1s",
        "num_gt_objects", "num_never_matched",
    }),
    "porr": frozenset({
        "porr_mean", "porr_num_events",
        "porr_skipped_no_visibility", "porr_n_size_bins",
    }),
}


def enabled_overall_groups(metric_names: List[str]) -> Set[str]:
    """Map config :attr:`EvalConfig.metrics` entries to :attr:`EvalReport.overall` keys.

    ``clear`` / ``identity`` align with TrackEval naming (``mota`` / ``idf1`` groups).
    """
    out: set[str] = set()
    for raw in metric_names:
        if raw == "clear":
            out.add("mota")
        elif raw == "identity":
            out.add("idf1")
        else:
            out.add(raw)
    return out


def iter_overall_scalar_kpis(
    overall: Dict[str, Dict[str, float]],
    enabled_groups: Set[str],
) -> Iterator[tuple[str, float]]:
    """Yield ``(series, value)`` pairs for :meth:`clearml.Logger.report_scalar`.

    *series* is ``"{group}/{key}"`` (e.g. ``mota/mota``, ``hota/hota``) so names are
    unique in the **Inference** scalar group across MOT metrics.
    Non-finite values are skipped.
    """
    for group in sorted(enabled_groups & set(overall.keys())):
        keys = _OVERALL_SCALAR_KEYS.get(group)
        if not keys:
            continue
        row = overall[group]
        for key in sorted(keys & row.keys()):
            val = float(row[key])
            if not math.isfinite(val):
                continue
            yield (f"{group}/{key}", val)


# ------------------------------------------------------------------
# ClearML reporter
# ------------------------------------------------------------------

@register_reporter("clearml")
class ClearMLReporter(Reporter):
    """Send evaluation results to the active ClearML task.

    Output in the ClearML web UI:

    * **SCALARS** -- overall (micro-aggregated ``all`` bin) KPIs under the
      title **Inference**, one series per ``{metric_group}/{field}`` (e.g.
      ``mota/mota``, ``hota/hota``). Use **Scalars** comparison across tasks
      to diff settings on the same named series.
    * **PLOTS** -- bar-charts (histogram mode) for slice breakdowns. Each
      headline metric is logged as its own **variant** (series name), so
      comparing two tasks overlays at most two traces per bar group.

      - *Per-Class Metrics*  -- headline metrics across classes.
      - *Per-Size Metrics*   -- headline metrics across size bins.
      - *{class}-Per-Size*   -- per-size breakdown for each class
        (requires both ``class_groups`` and ``size_bins`` in the config).

    * **Optional slice scalars** -- when ``EvalConfig.clearml_slice_scalars``
      is true, per-bin headline values are also logged under titles such as
      ``PerClass/{metric}`` for **Scalars → Values** CSV-style comparison.

    * **TABLES** -- count summaries (not scalars): *Overall Counts* and
      per-class count tables; PORR size×time tables when ``porr`` is enabled.

    Bar-chart headline metrics follow :attr:`PLOT_METRICS`.
    """

    OVERALL_SCALAR_FIELDS: List[str] = [
        "num_tp", "num_fp", "num_fn", "num_idsw", "frag",
        "mt", "pt", "ml", "num_gt_objects", "num_gt",
        "num_frames", "far",
    ]

    PER_BIN_TABLE_FIELDS: List[str] = [
        "num_gt_objects", "num_gt",
        "num_tp", "num_fp", "num_fn", "num_idsw", "frag", "mt", "pt", "ml",
        "num_frames", "far",
    ]

    PLOT_METRICS: List[str] = [
        "hota", "idf1", "coverage", "pd", "id_instability", "far",
        "tid_ratio_immediate",
    ]

    PLOT_METRIC_SOURCE: Dict[str, tuple[str, str]] = {
        "far": ("mota", "far"),
        "tid_ratio_immediate": ("realtime_kpi", "tid_ratio_immediate"),
    }

    def report(self, eval_report: EvalReport, output_dir: Path) -> None:
        try:
            from clearml import Task
        except ImportError:
            logger.warning("clearml package not installed; skipping ClearML reporter.")
            return

        task = Task.current_task()
        if task is None:
            logger.warning("No active ClearML task; skipping ClearML reporter.")
            return

        cl = task.get_logger()
        cfg = eval_report.config
        agg = eval_report.aggregated_bins

        self._report_inference_scalars(cl, eval_report)

        plot_active = [m for m in cfg.metrics if m in self.PLOT_METRICS]
        if "mota" in cfg.metrics and "far" in self.PLOT_METRICS and "far" not in plot_active:
            plot_active.append("far")

        class_keys = [k for k in (cfg.class_groups or {}) if k in agg]
        size_keys = [k for k in (cfg.size_bins or {}) if k in agg]
        class_order = list((cfg.class_groups or {}).keys())

        # --- 1. Summary count tables ---
        has_mota = any(m in ("mota", "clear") for m in cfg.metrics)
        if has_mota:
            self._report_scalar_tables(
                cl, eval_report.overall, agg, class_order,
            )

        # --- 2. Overview bar charts (cross-class / cross-size) ---
        if class_keys:
            self._report_bar_chart(
                cl, agg, class_keys, plot_active,
                title="2a. Per-Class Metrics", xaxis="Class",
            )
        if size_keys:
            self._report_bar_chart(
                cl, agg, size_keys, plot_active,
                title="2b. Per-Size Metrics", xaxis="Size Bin",
            )

        # --- 3. Per-class detail charts ---
        if class_keys and size_keys:
            self._report_per_class_size_charts(
                cl, agg, class_order, size_keys, plot_active,
            )

        # --- 4. Density bar chart ---
        density_overall: Dict[str, Any] | None = None
        density_agg = eval_report.density_aggregated
        if density_agg:
            density_keys = list(density_agg.keys())
            density_overall = {
                dk: density_agg[dk]["all"]
                for dk in density_keys
                if "all" in density_agg[dk]
            }
            if density_overall:
                self._report_bar_chart(
                    cl, density_overall, list(density_overall.keys()),
                    plot_active,
                    title="2c. Per-Density Metrics", xaxis="Scene Density",
                )

        if cfg.clearml_slice_scalars:
            self._report_clearml_slice_scalars(
                cl, agg, class_keys, size_keys, class_order, plot_active,
                density_overall,
            )

        if "porr" in cfg.metrics:
            self._report_porr_tables(cl, eval_report)

        cl.flush()
        logger.info("ClearML reporting complete.")

    # -- helpers -------------------------------------------------------

    def _report_inference_scalars(self, cl, eval_report: EvalReport) -> None:
        """Log micro-aggregated overall KPIs as ClearML Scalars (Compare-friendly)."""
        enabled = enabled_overall_groups(eval_report.config.metrics)
        for series, val in iter_overall_scalar_kpis(eval_report.overall, enabled):
            cl.report_scalar("Inference", series, val, iteration=0)

    def _report_scalar_tables(
        self, cl, overall: Dict, agg: Dict,
        class_order: List[str],
    ) -> None:
        """Report count metrics as numbered ClearML tables, one per group.

        Creates (in order):
        * **1a. Overall Counts** — single-row table with aggregate counts.
        * **1b. Counts - {class}** — one table per class group showing
          per-size-bin breakdowns, ordered by the config's class order.
        """
        import pandas as pd

        mota_vals = overall.get("mota", {})

        overall_row = {}
        for f in self.OVERALL_SCALAR_FIELDS:
            if f not in mota_vals:
                continue
            v = mota_vals[f]
            overall_row[f] = round(v, 6) if f == "far" else int(v)
        if overall_row:
            df = pd.DataFrame([overall_row])
            cl.report_table(
                "1a. Overall Counts", "summary",
                iteration=0, table_plot=df,
            )

        grouped = dict(_group_slices_by_class(agg))
        for idx, cls_name in enumerate(class_order):
            items = grouped.get(cls_name)
            if not items:
                continue
            rows: list[dict] = []
            for bin_name, bin_result in items:
                vals = bin_result.metric_values.get("mota", {})
                if not vals:
                    continue
                size_label = (
                    bin_name.split(" / ", 1)[1]
                    if " / " in bin_name else bin_name
                )
                row: dict = {"size_bin": size_label}
                for f in self.PER_BIN_TABLE_FIELDS:
                    if f in vals:
                        row[f] = round(vals[f], 6) if f == "far" else int(vals[f])
                rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                letter = chr(ord("b") + idx)
                cl.report_table(
                    f"1{letter}. Counts - {cls_name}", "by_size",
                    iteration=0, table_plot=df,
                )

    @staticmethod
    def _report_bar_chart(
        cl, agg: Dict, keys: List[str],
        active: List[str], *, title: str, xaxis: str,
    ) -> None:
        """One ``report_histogram`` per headline metric (ClearML variant = *mn*)."""
        import numpy as np

        for mn in active:
            group, key = ClearMLReporter.PLOT_METRIC_SOURCE.get(
                mn, (mn, mn),
            )
            vals = [
                float(agg[k].metric_values.get(group, {}).get(key, 0.0))
                for k in keys
            ]
            row = np.array([vals], dtype=float)
            cl.report_histogram(
                title=title,
                series=mn,
                values=row,
                labels=[mn],
                xlabels=keys,
                xaxis=xaxis,
                yaxis="Score",
                mode="group",
                extra_layout={"title": f"{title} — {mn}"},
            )

    def _report_porr_tables(self, cl, eval_report: EvalReport) -> None:
        """Log PORR matrices as ClearML tables (global ``all`` and per-class bins)."""
        from evaluation.filtering import porr_metric_row_slug, size_bins_for_porr
        from evaluation.metrics.porr import TIME_COL_LABELS, porr_table_arrays

        porr_bins = size_bins_for_porr(eval_report.config.size_bins)
        row_labels = [b.name for b in porr_bins]
        row_slugs = [porr_metric_row_slug(b.name) for b in porr_bins]
        time_headers = [str(t) for t in TIME_COL_LABELS]

        bins_with_porr: List[tuple[str, Dict[str, float]]] = []
        for bname, br in eval_report.aggregated_bins.items():
            if "porr" in br.metric_values:
                bins_with_porr.append((bname, br.metric_values["porr"]))

        if not bins_with_porr:
            return

        import pandas as pd

        for plot_idx, (bin_name, porr_flat) in enumerate(bins_with_porr):
            rates, counts = porr_table_arrays(porr_flat, row_slugs)
            suffix = "" if bin_name == "all" else f" [{bin_name}]"
            label_tag = "a" if bin_name == "all" else chr(ord("a") + plot_idx)
            table_title = f"4{label_tag}. PORR table{suffix}"

            try:
                table_data = []
                for i, label in enumerate(row_labels):
                    row: dict = {"size \\ time (s)": label}
                    for j, th in enumerate(time_headers):
                        c = int(counts[i, j])
                        if c > 0:
                            row[th] = round(float(rates[i, j]), 4)
                        else:
                            row[th] = None
                    table_data.append(row)
                df = pd.DataFrame(table_data).set_index("size \\ time (s)")
                cl.report_table(
                    table_title, "porr",
                    iteration=0, table_plot=df,
                )
            except Exception as exc:
                logger.warning("ClearML PORR table report failed (%s): %s", bin_name, exc)

    def _get_plot_value(self, bin_result, mn: str) -> float:
        """Resolve metric value for bar charts, using PLOT_METRIC_SOURCE if set."""
        group, key = self.PLOT_METRIC_SOURCE.get(mn, (mn, mn))
        return float(bin_result.metric_values.get(group, {}).get(key, 0.0))

    def _report_per_class_size_charts(
        self, cl, agg: Dict,
        class_order: List[str], size_keys: List[str],
        active: List[str],
    ) -> None:
        """One bar chart per class; one ClearML variant per headline metric."""
        import numpy as np

        for idx, cls_name in enumerate(class_order):
            slice_keys = [f"{cls_name} / {sb}" for sb in size_keys]
            present = [sk for sk in slice_keys if sk in agg]
            if not present:
                continue

            size_labels = [sk.split(" / ", 1)[1] for sk in present]
            letter = chr(ord("a") + idx)
            chart_title = f"3{letter}. {cls_name} - Per-Size"

            for mn in active:
                vals = [self._get_plot_value(agg[sk], mn) for sk in present]
                row = np.array([vals], dtype=float)
                cl.report_histogram(
                    title=chart_title,
                    series=mn,
                    values=row,
                    labels=[mn],
                    xlabels=size_labels,
                    xaxis="Size Bin",
                    yaxis="Score",
                    mode="group",
                    extra_layout={"title": f"{chart_title} — {mn}"},
                )

    def _report_clearml_slice_scalars(
        self,
        cl,
        agg: Dict[str, Any],
        class_keys: List[str],
        size_keys: List[str],
        class_order: List[str],
        plot_active: List[str],
        density_overall: Dict[str, Any] | None,
    ) -> None:
        """Log per-slice headline metrics as scalars for tabular task comparison."""
        if class_keys:
            self._report_slice_scalars_for_bins(
                cl, "PerClass", agg, class_keys, plot_active,
            )
        if size_keys:
            self._report_slice_scalars_for_bins(
                cl, "PerSize", agg, size_keys, plot_active,
            )
        if density_overall:
            self._report_slice_scalars_for_bins(
                cl, "PerDensity", density_overall,
                list(density_overall.keys()), plot_active,
            )
        if class_keys and size_keys:
            self._report_per_class_size_slice_scalars(
                cl, agg, class_order, size_keys, plot_active,
            )

    def _report_slice_scalars_for_bins(
        self,
        cl,
        title_prefix: str,
        bins_map: Dict[str, Any],
        bin_keys: List[str],
        plot_active: List[str],
    ) -> None:
        for mn in plot_active:
            group, key = self.PLOT_METRIC_SOURCE.get(mn, (mn, mn))
            title = f"{title_prefix}/{mn}"
            for bk in bin_keys:
                br = bins_map.get(bk)
                if br is None:
                    continue
                raw = br.metric_values.get(group, {}).get(key)
                if raw is None:
                    continue
                val = float(raw)
                if not math.isfinite(val):
                    continue
                cl.report_scalar(title, str(bk), val, iteration=0)

    def _report_per_class_size_slice_scalars(
        self,
        cl,
        agg: Dict[str, Any],
        class_order: List[str],
        size_keys: List[str],
        plot_active: List[str],
    ) -> None:
        for cls_name in class_order:
            for mn in plot_active:
                title = f"PerClassSize/{cls_name}/{mn}"
                for sb in size_keys:
                    sk = f"{cls_name} / {sb}"
                    if sk not in agg:
                        continue
                    val = self._get_plot_value(agg[sk], mn)
                    if not math.isfinite(val):
                        continue
                    cl.report_scalar(title, sb, val, iteration=0)
