import pandas as pd
import numpy as np
from .mixins import PlottingSamplingMixin

class SamplingMetricsSummarizer(PlottingSamplingMixin):
    def __init__(self):
        pass
    # Added functions to summarize means and SEs of metrics across folds for each sampling
    # For each sampling

    # --- Helper: compute mean, std, sem, n across folds for one metric frame -----
    def _summarize_frame(self, df_bins_by_parts):
        """
        df_bins_by_parts: DataFrame (rows=bins, cols=parts)
        Returns a DataFrame with index=cell_type and columns ['mean','std','sem','n'].
        """
        mean = df_bins_by_parts.mean(axis=0, skipna=True)
        std = df_bins_by_parts.std(axis=0, ddof=1, skipna=True)
        n = df_bins_by_parts.count(axis=0)  # number of bins contributing per part
        sem = std / np.sqrt(n.clip(lower=1))
        out = pd.DataFrame({"mean": mean, "std": std, "sem": sem, "n": n})
        out.index.name = "cell_type"
        return out

    # --- Summarize all metrics for one cell type --------------------------------
    def summarize_one_celltype(self, celltype_name, metrics_by_bin, metric_names=None, run_id=None):
        """
        Build a long-form summary for one cell type:
        rows = parts, columns = ['cell_type','metric','part','mean','std','sem','n', 'run_id'].
        """
        # Build frames for all metrics (bins x parts)
        frames = self.build_metric_frames(metrics_by_bin)

        if metric_names is None:
            metric_names = list(frames.keys())

        records = []
        for m in metric_names:
            df = frames[m]  # bins x cell_type
            summary = self._summarize_frame(df)  # index=cell_type
            for celltype, row in summary.iterrows():
                records.append({
                    "cell_type": celltype_name,
                    "metric": m,
                    "celltype": celltype,
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "sem": float(row["sem"]),
                    "n": int(row["n"]),
                    "run_id": run_id
                })

        return pd.DataFrame.from_records(records)

    # --- Summarize multiple cell types at once ----------------------------------
    def summarize_metrics(self, metrics_dict, metric_names=None, run_id=None):
        """
        metrics_dict: {cell_type_name: metrics_by_bin_dict}
        Returns a long-form DataFrame over all cell types:
          columns = ['cell_type','metric','part','mean','std','sem','n','run_id'].
        """
        dfs = []
        for ct_name, m_by_bin in metrics_dict.items():
            dfs.append(self.summarize_one_celltype(ct_name, m_by_bin, metric_names=metric_names, run_id=run_id))
        return pd.concat(dfs, axis=0, ignore_index=True)

    # summarize_across_samplings:
    # - Computes between-sampling mean, std, sem for each (cell_type, metric, celltype)
    # - Also returns optional within-sampling summaries (averages and pooled)
    def summarize_across_samplings(self, summary_df_dict, metric_names=None, include_within=True):
        """
        Parameters
        ----------
        summary_df_dict : dict[sampling_id, pd.DataFrame]
            Each DataFrame must contain columns:
            ['cell_type', 'metric', 'celltype', 'mean', 'std', 'sem', 'n'].
            An optional 'run_id' column is ignored.
        metric_names : list[str] or None
            If provided, only include these metrics.
        include_within : bool
            If True, add avg_within_std/sem and pooled_within_std/sem_from_within.

        Returns
        -------
        pd.DataFrame
            Columns:
            ['cell_type', 'metric', 'celltype',
            'mean_sampling', 'std_sampling', 'sem_sampling', 'n_sampling']
            plus optional within-sampling summaries if include_within=True.
        """

        # Concatenate and tag rows with sampling_id
        frames = []
        for sampling_id, df in summary_df_dict.items():
            df2 = df.copy()
            df2["sampling_id"] = sampling_id
            frames.append(df2)
        all_df = pd.concat(frames, axis=0, ignore_index=True)

        # Optional filter by metric
        if metric_names is not None:
            all_df = all_df[all_df["metric"].isin(metric_names)].copy()

        # Ensure numeric types
        for col in ["mean", "std", "sem", "n"]:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

        # Group by (cell_type, metric, celltype)
        group_cols = ["cell_type", "metric", "celltype"]
        groups = all_df.groupby(group_cols, dropna=False)

        def summarize_group(df):
            # Per-sampling means (between-sampling)
            x = df["mean"].to_numpy(dtype=float)
            k = np.count_nonzero(~np.isnan(x))  # number of samplings
            mean_sampling = float(np.nanmean(x)) if k > 0 else np.nan
            std_sampling = float(np.nanstd(x, ddof=1)) if k > 1 else 0.0  # between-sampling std
            sem_sampling = (std_sampling / np.sqrt(k)) if k > 0 else np.nan

            result = {
                "mean_sampling": mean_sampling,
                "std_sampling": std_sampling,
                "sem_sampling": sem_sampling,
                "n_sampling": int(k),
            }

            if include_within:
                # Descriptive averages of within-sampling variability
                avg_within_std = float(df["std"].mean()) if len(df) else np.nan
                avg_within_sem = float(df["sem"].mean()) if len(df) else np.nan

                # Pooled within-sampling std across samplings (weighted by df = n_i - 1)
                n_i = df["n"].to_numpy(dtype=float)
                s_i = df["std"].to_numpy(dtype=float)
                df_sum = np.nansum(np.clip(n_i - 1, a_min=0, a_max=None))
                num = np.nansum(np.clip(n_i - 1, a_min=0, a_max=None) * (s_i ** 2))
                pooled_within_std = float(np.sqrt(num / df_sum)) if df_sum > 0 else np.nan

                # SEM of the mean-of-means due to within-sampling uncertainty only:
                # var(mean_of_means) = (1/k^2) * sum_i (std_i^2 / n_i), assuming independence
                valid = (n_i > 0) & np.isfinite(s_i)
                denom = (float(k) ** 2) if k > 0 else np.nan
                sem_from_within = np.sqrt(np.nansum((s_i[valid] ** 2) / n_i[valid]) / denom) if k > 0 else np.nan

                result.update({
                    "avg_within_std": avg_within_std,
                    "avg_within_sem": avg_within_sem,
                    "pooled_within_std": pooled_within_std,
                    "sem_from_within": float(sem_from_within),
                })

            return pd.Series(result)

        out = groups.apply(summarize_group).reset_index()
        return out