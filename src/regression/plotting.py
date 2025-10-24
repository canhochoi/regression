import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from .mixins import PlottingSamplingMixin

class Plotting(PlottingSamplingMixin):
    def __init__(self):
        pass
    
    # # -- Stack one metric across bins into a DataFrame (rows=bins, cols=parts)
    # def stack_metric(self, metrics_by_bin, key):
    #     rows = {}
    #     for b, d in metrics_by_bin.items():
    #         s = d[key]
    #         # Accept Series or 1D np.ndarray; convert to Series if needed
    #         if isinstance(s, np.ndarray):
    #             s = pd.Series(s, index=d.get("Y_test_comp_df").columns)
    #         rows[b] = s
    #     return pd.DataFrame(rows).T  # bins as rows
    
    
    # # -- Build all four metric DataFrames at once
    # def build_metric_frames(self, metrics_by_bin):
    #     return {
    #         "cors_comp": self.stack_metric(metrics_by_bin, "cors_comp"),
    #         "cors_clr_comp": self.stack_metric(metrics_by_bin, "cors_clr_comp"),
    #         "Rsq_comp": self.stack_metric(metrics_by_bin, "Rsq_comp"),
    #         "Rsq_clr_comp": self.stack_metric(metrics_by_bin, "Rsq_clr_comp"),
    #     }
    
    # -- Align two DataFrames (same metric) on common bins and parts
    def align_two(self, df_x, df_y):
        bins = df_x.index.intersection(df_y.index)
        parts = df_x.columns.intersection(df_y.columns)
        return df_x.loc[bins, parts], df_y.loc[bins, parts], bins, parts
    
    # -- Combine two metric frames into a single wide frame with MultiIndex columns
    def combine_two_metrics(self, df_x, df_y, name_x="HSC", name_y="HSC_B"):
        df_x_a, df_y_a, bins, parts = self.align_two(df_x, df_y)
        combined = pd.concat({name_x: df_x_a, name_y: df_y_a}, axis=1)
        return combined  # columns: MultiIndex (name, part), rows: bins

    # Combine two dataframes
    def combined_metric_df(self, frames_dict, metric):
        celltype_list = list(frames_dict.keys())
        metric_A = frames_dict[celltype_list[0]][metric]
        metric_B = frames_dict[celltype_list[1]][metric]
        
        combined_metric = self.combine_two_metrics(metric_A, metric_B, name_x = celltype_list[0], name_y = celltype_list[1])
    
        return combined_metric
    
    # -- Scatter plot per part: x = metric for group X, y = metric for group Y
    def plot_scatter_per_part(self, combined, name_x="HSC", name_y="HSC_B", metric_name="cors_comp"):
        """
        combined: output of combine_two_metrics (MultiIndex columns: (group, part))
        # Plots a grid of scatter plots (one per part) with y=x reference.
        # """
        parts = combined.columns.get_level_values(1).unique()
        n_parts = len(parts)
        ncols = min(4, n_parts)
        nrows = int(np.ceil(n_parts / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
    
        for i, part in enumerate(parts):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            x = combined[(name_x, part)]
            y = combined[(name_y, part)]
            sns.scatterplot(x=x, y=y, ax=ax)
            # y=x line
            lim = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lim, lim, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"{metric_name}: {part}")
            ax.set_xlabel(name_x)
            ax.set_ylabel(name_y)
    
        # Hide unused subplots
        for j in range(i+1, nrows*ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
    
        fig.tight_layout()
        return fig

    # check if adding terminal cells help cell type proportion prediction
    # e.g. HSC_MPP vs HSC_MPP_B
    # -- Align two metric DataFrames (rows=bins, cols=parts)
    def _align_two(self, df_x, df_y):
        bins = df_x.index.intersection(df_y.index)
        parts = df_x.columns.intersection(df_y.columns)
        return df_x.loc[bins, parts], df_y.loc[bins, parts], bins, parts
    
    # -- Compute mean and standard error per part across bins
    def _mean_sem(self, df_aligned):
        """
        df_aligned: DataFrame (rows=bins, cols=parts)
        Returns:
          mean: Series (parts)
          sem:  Series (parts), computed as sample std / sqrt(n)
          n:    Series (parts), number of non-NA bins per part
        """
        mean = df_aligned.mean(axis=0, skipna=True)
        std = df_aligned.std(axis=0, ddof=1, skipna=True)
        n = df_aligned.count(axis=0)
        sem = std / np.sqrt(n.clip(lower=1))
        return mean, sem, n
    
    # -- Prepare scatter data for two groups (HSC vs HSC_B)
    def prepare_scatter_with_se(self, frames_dict, metric_name):
        """
        df_x, df_y: DataFrames for the same metric (rows=bins, cols=parts)
        Returns:
          data_df: DataFrame with columns:
                   ["part", f"{name_x}_mean", f"{name_x}_sem", f"{name_y}_mean", f"{name_y}_sem",
                    f"{name_x}_n", f"{name_y}_n"]
        """
        name_x = list(frames_dict.keys())[0]
        name_y = list(frames_dict.keys())[1]
        df_x = frames_dict[name_x][metric_name]
        df_y = frames_dict[name_y][metric_name]
        df_x_a, df_y_a, bins, parts = self._align_two(df_x, df_y)
        mx, sx, nx = self._mean_sem(df_x_a)
        my, sy, ny = self._mean_sem(df_y_a)
        data = pd.DataFrame({
            "part": parts,
            f"{name_x}_mean": mx.values,
            f"{name_x}_sem": sx.values,
            f"{name_x}_n": nx.values,
            f"{name_y}_mean": my.values,
            f"{name_y}_sem": sy.values,
            f"{name_y}_n": ny.values,
        }, index=parts)
        return data
    
    # -- Plot scatter with horizontal/vertical error bars per part
    def plot_scatter_with_se(self, data_df, frames_dict, metric_name,
                         annotate=True, refline=True, ax=None):
        """
        data_df: output of prepare_scatter_with_se
        frames_dict: {"name_x": frames_X, "name_y": frames_Y}, only used to label axes
        metric_name: str, e.g., "cors_comp"
        If ax is provided, draw into it; else create a new figure+axes.
        """
        name_x, name_y = list(frames_dict.keys())
        x_mean = data_df[f"{name_x}_mean"].to_numpy()
        y_mean = data_df[f"{name_y}_mean"].to_numpy()
        x_err = data_df[f"{name_x}_sem"].to_numpy()
        y_err = data_df[f"{name_y}_sem"].to_numpy()
        parts = data_df["part"].to_numpy()

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots(figsize=(6, 6))

        ax.errorbar(x_mean, y_mean, xerr=x_err, yerr=y_err, fmt='o', ecolor='gray',
                    elinewidth=1.0, capsize=3, capthick=1.0, color='C0')

        if annotate:
            for xi, yi, label in zip(x_mean, y_mean, parts):
                ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 2), fontsize=8)

        ax.set_xlabel(f"{metric_name} ({name_x})")
        ax.set_ylabel(f"{metric_name} ({name_y})")
        ax.set_title(f"{metric_name}: {name_x} vs {name_y} (mean ± SE across bins)")

        if refline:
            lo = min(np.nanmin(x_mean), np.nanmin(y_mean))
            hi = max(np.nanmax(x_mean), np.nanmax(y_mean))
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)

        ax.grid(True, alpha=0.2)

        # Return fig, ax (fig is either created here or the one owning ax)
        fig = created_fig if created_fig is not None else ax.figure
        fig.tight_layout()
        return fig, ax
    
    # --- New: panel function to combine multiple metrics in one figure ----------
    def scatter_two_celltypes_panel(self, celltype_x, metrics_by_bin_x,
                                    celltype_y, metrics_by_bin_y,
                                    metric_names=("cors_comp", "cors_clr_comp", "Rsq_comp", "Rsq_clr_comp"),
                                    ncols=2, figsize=None, annotate=True, refline=True):
        """
        Create a panel figure with one subplot per metric in metric_names.
        """
        n = len(metric_names)
        nrows = (n + ncols - 1) // ncols
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for i, m in enumerate(metric_names):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            # Draw into this axes
            self.scatter_two_celltypes_bins(celltype_x=celltype_x,
                                            metrics_by_bin_x=metrics_by_bin_x,
                                            celltype_y=celltype_y,
                                            metrics_by_bin_y=metrics_by_bin_y,
                                            metric_name=m,
                                            ax=ax,
                                            annotate=annotate,
                                            refline=refline)
            ax.set_title(f"{m}: {celltype_x} vs {celltype_y}")

        # Hide any unused subplots
        for j in range(i + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        fig.tight_layout()
        return fig, axes

    # --- New: panel function to combine multiple metrics and multiple pairs ------
    def scatter_design_matrices(self, metrics_dict, metric_names,
                            pairs=None, ncols=3, figsize=None,
                            annotate=True, refline=True):
        """
        Create a panel of subplots. Each subplot is a scatter-with-SE for one
        metric and one pair of cell types from metrics_dict.

        Parameters
        ----------
        metrics_dict : dict[str, dict]
            {"HSC_MPP": metrics_by_bin_HSC, "B": metrics_by_bin_B, "HSC_MPP_B": metrics_by_bin_HSC_B}
        metric_names : list[str]
            e.g., ["cors_comp", "cors_clr_comp", "Rsq_comp", "Rsq_clr_comp"]
        pairs : list[tuple[str, str]] or None
            If None, use all unique pairs from the keys via combinations.
            Example: [("HSC_MPP", "HSC_MPP_B"), ("B", "HSC_MPP_B"), ("HSC_MPP", "B")]
        ncols : int
            Number of columns in the panel.
        figsize : tuple or None
            Figure size; if None, chosen based on ncols and number of subplots.
        annotate, refline : bool
            Passed through to scatter_two_celltypes_bins.

        Returns
        -------
        fig, axes
        """
        # Build pairs if not provided
        keys = list(metrics_dict.keys())
        if pairs is None:
            pairs = list(itertools.combinations(keys, 2))

        # Total subplots = number of metrics * number of pairs
        n_subplots = len(metric_names) * len(pairs)
        nrows = n_subplots // ncols
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        # Iterate over product (metric, pair) and draw each into its Axes
        for idx, (metric, (celltype_x, celltype_y)) in enumerate(itertools.product(metric_names, pairs)):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            self.scatter_two_celltypes_bins(
                celltype_x=celltype_x,
                metrics_by_bin_x=metrics_dict[celltype_x],
                celltype_y=celltype_y,
                metrics_by_bin_y=metrics_dict[celltype_y],
                metric_name=metric,
                ax=ax,
                annotate=annotate,
                refline=refline
            )
            ax.set_title(f"{metric}: {celltype_x} vs {celltype_y}", fontsize=10)

        # Hide any unused axes
        for j in range(idx + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        fig.tight_layout()
        return fig, axes

    def scatter_two_celltypes_bins(self, celltype_x, metrics_by_bin_x, celltype_y, metrics_by_bin_y, metric_name, ax=None, annotate=True, refline=True):
        # better version
        metrics_dict = {celltype_x: metrics_by_bin_x,
                        celltype_y: metrics_by_bin_y}
        
        frames_dict = {celltype: self.build_metric_frames(metrics) for celltype, metrics in metrics_dict.items()}
        
        data_cors_clr = self.prepare_scatter_with_se(frames_dict, metric_name = metric_name)
        
        fig, ax = self.plot_scatter_with_se(data_cors_clr, frames_dict, metric_name = metric_name,
                                            annotate=annotate, refline=refline, ax=ax)

        return fig, ax


    # plot test vs observed for each fold across cell types

    def plot_test_vs_obs_grid(self, Y_test_df, Y_obs_df, ncols=4, figsize=None,
                          sharex=False, sharey=False, refline=True, title_prefix=""):
        """
        Plot a grid of scatter plots: test vs obs for each column.
        Each subplot title includes Pearson correlation and R^2.
    
        Parameters
        ----------
        Y_test_df : pandas.DataFrame
            Predicted/computed compositions (rows=samples, cols=cell types).
        Y_obs_df : pandas.DataFrame
            Observed compositions (same shape/columns as Y_test_df).
        ncols : int
            Number of columns in the subplot grid.
        figsize : tuple or None
            Figure size; if None, chosen based on number of subplots.
        sharex, sharey : bool
            Share axes across subplots.
        refline : bool
            Draw y = x reference line in each subplot.
        title_prefix : str
            Optional prefix for each subplot title.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of Axes
        metrics_df : pandas.DataFrame
            Per-column metrics with columns ["pearson_r", "r_squared", "n_pairs"].
        """
        # -- Align columns and index
        common_cols = Y_test_df.columns.intersection(Y_obs_df.columns)
        Yt = Y_test_df[common_cols]
        Yo = Y_obs_df[common_cols]
    
        # -- Grid layout
        n_parts = len(common_cols)
        nrows = int(np.ceil(n_parts / ncols))
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
    
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)
    
        # -- Collect metrics
        metrics = []
    
        for i, part in enumerate(common_cols):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
    
            # Pairwise drop NA for this part
            df_pair = pd.DataFrame({"test": Yt[part], "obs": Yo[part]}).dropna()
            x = df_pair["test"].to_numpy()
            y = df_pair["obs"].to_numpy()
            n = len(df_pair)
    
            # Scatter
            ax.scatter(x, y, s=20, alpha=0.7)
    
            # Pearson correlation (pairwise)
            if n >= 2:
                r_pearson = df_pair["test"].corr(df_pair["obs"])
            else:
                r_pearson = np.nan
    
            # R^2 (against obs mean baseline)
            if n >= 2:
                sse = np.sum((x - y) ** 2)
                y_mean = np.mean(y)
                sst = np.sum((y - y_mean) ** 2)
                r_sq = (1.0 - sse / sst) if sst > 0 else np.nan
            else:
                r_sq = np.nan
    
            # Reference line y=x
            if refline and n > 0:
                lo = min(np.min(x), np.min(y))
                hi = max(np.max(x), np.max(y))
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    
            ax.set_title(f"{title_prefix}{part} (r={r_pearson:.3f}, R²={r_sq:.3f})", fontsize=9)
            ax.set_xlabel("test")
            ax.set_ylabel("obs")
            ax.grid(True, alpha=0.2)
    
            metrics.append({"part": part, "pearson_r": r_pearson, "r_squared": r_sq, "n_pairs": n})
    
        # Hide unused axes
        for j in range(i + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
    
        fig.tight_layout()
    
        metrics_df = pd.DataFrame(metrics).set_index("part")
        return fig, axes, metrics_df


    # -- Compute per-part metric across folds -------------------------------------

    def compute_fold_metrics(self, metrics_by_bin, metric="corr"):
        """
        Build a long-form DataFrame of a per-part metric across folds.
    
        Parameters
        ----------
        metrics_by_bin : dict[int, dict]
            For each fold key, contains:
              - "Y_test_comp_df": DataFrame (rows=test samples in fold, cols=parts)
              - "Y_obs_comp_df" : DataFrame (same shape/columns)
        metric : {"corr", "rsq"}
            Metric to compute per part within each fold:
              - "corr": Pearson correlation across test samples in that fold
              - "rsq" : R^2 = 1 - SSE/SST across test samples in that fold
    
        Returns
        -------
        df_long : pandas.DataFrame
            Columns: ["fold", "part", "value"]
            One row per (fold, part).
        """
        records = []
    
        for fold, d in metrics_by_bin.items():
            Yt = d["Y_test_comp_df"]
            Yo = d["Y_obs_comp_df"]
    
            # Align columns (parts)
            parts = Yt.columns.intersection(Yo.columns)
    
            for part in parts:
                # Pairwise drop NA within this fold and part
                pair = pd.DataFrame({"test": Yt[part], "obs": Yo[part]}).dropna()
                x = pair["test"].to_numpy()
                y = pair["obs"].to_numpy()
                n = len(pair)
    
                if n < 2:
                    val = np.nan
                else:
                    if metric == "Correlation":
                        # Pearson correlation
                        val = pair["test"].corr(pair["obs"])
                    elif metric == "Rsquare":
                        # R^2 against mean baseline of observed values
                        sse = np.sum((x - y) ** 2)
                        y_mean = np.mean(y)
                        sst = np.sum((y - y_mean) ** 2)
                        val = (1.0 - sse / sst) if sst > 0 else np.nan
                    else:
                        raise ValueError(f"Unknown metric={metric!r}. Use 'corr' or 'rsq'.")
    
                records.append({"fold": fold, "part": part, "value": float(val)})
    
        df_long = pd.DataFrame(records)
        return df_long


    # -- Plot: box plot per part with scatter dots per fold -----------------------
    
    def plot_box_scatter(self, df_long, metric_label="Correlation",
                         jitter=0.25, hue=None, palette="tab10"):
        """
        Plot box plot per part with scattered per-fold dots overlaid.
    
        Parameters
        ----------
        df_long : pandas.DataFrame
            Long-form DataFrame with columns ["fold", "part", "value"].
        metric_label : str
            Y-axis label and title suffix (e.g., "Correlation", "R^2").
        jitter : float
            Horizontal jitter for scatter points to avoid overlap.
        hue : str or None
            Column name to color the scatter points (e.g., "fold"); if None, no hue.
        palette : str or list
            Seaborn palette for scatter when hue is used.
    
        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        parts = df_long["part"].unique().tolist()
        n_parts = len(parts)
        # Auto figure width based on number of parts
        fig_w = max(6, 0.6 * n_parts)
        fig_h = 6
    
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
        # Boxplot of distribution across folds per part
        sns.boxplot(data=df_long, x="part", y="value", ax=ax,
                    color="lightgray", showcaps=True, boxprops={"alpha": 0.6})
    
        # Scatter per fold per part
        if hue is None:
            # No hue: plain scatter with jitter
            # Compute x positions per category
            x_positions = {p: i for i, p in enumerate(parts)}
            xs = df_long["part"].map(x_positions).to_numpy().astype(float)
            rng = np.random.default_rng(0)
            xs = xs + rng.uniform(-jitter, jitter, size=len(xs))
            ax.scatter(xs, df_long["value"], s=30, alpha=0.8, color="C0", zorder=3)
        else:
            # Use seaborn stripplot with hue
            sns.stripplot(data=df_long, x="part", y="value", hue=hue,
                          ax=ax, jitter=jitter, dodge=False, palette=palette, zorder=3)
    
            # Move legend outside
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    
        ax.set_xlabel("Cell type (part)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} per cell type across folds (box: distribution, dots: folds)")
        ax.grid(True, axis="y", alpha=0.2)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
        fig.tight_layout()
        return fig, ax

    def distribution_across_folds(self, metrics_by_bin, metric_name):
        df_metric = self.compute_fold_metrics(metrics_by_bin, metric = metric_name)
        fig, ax = self.plot_box_scatter(df_metric, metric_label = metric_name, hue="fold")
        return fig, ax
    
    # --- For sampling 
    # --- Plot scatter of different metrics for different cell types based on samplings ----------------
    def plot_sampling_summary_scatter_panel(self,
        df_across,
        metric_names,
        pairs=None,
        error_col="std_sampling",
        annotate=True,
        refline=True,
        ncols=None,
        figsize=None,
        sharex=False,
        sharey=False,
    ):
        """
        Parameters
        ----------
        df_across : pd.DataFrame
            Must contain columns:
            ['cell_type','metric','celltype','mean_sampling','std_sampling','sem_sampling','n_sampling'].
        metric_names : list[str]
            Metrics to plot as rows.
        pairs : list[tuple[str, str]] or None
            Pairs of cell_type to compare as columns. If None, use all combinations.
        error_col : {'std_sampling', 'sem_sampling'}
            Which column to use for error bars.
        annotate : bool
            Annotate each point with its celltype (part).
        refline : bool
            Draw y = x reference line in each subplot.
        ncols : int or None
            Number of columns (pairs) in the panel. Defaults to len(pairs).
        figsize : tuple or None
            Figure size; if None, chosen based on layout.
        sharex, sharey : bool
            Share axes across subplots.

        Returns
        -------
        fig, axes
        """
        # Validate error_col
        if error_col not in {"std_sampling", "sem_sampling"}:
            raise ValueError("error_col must be 'std_sampling' or 'sem_sampling'.")

        # Build pairs from available cell_type keys if not provided
        cell_types = sorted(df_across["cell_type"].unique().tolist())
        if pairs is None:
            pairs = list(itertools.combinations(cell_types, 2))

        # Layout: rows = metrics, cols = pairs
        nrows = len(metric_names)
        ncols = ncols or len(pairs)
        if figsize is None:
            figsize = (6 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)

        # Pre-split df by metric for speed
        df_by_metric = {m: df_across[df_across["metric"] == m].copy() for m in metric_names}

        for i, m in enumerate(metric_names):
            dfm = df_by_metric[m]
            for j, (ct_x, ct_y) in enumerate(pairs):
                ax = axes[i][j]

                # Data for each cell_type, indexed by "celltype" (part)
                dx = (
                    dfm[dfm["cell_type"] == ct_x][["celltype", "mean_sampling", error_col]]
                    .dropna(subset=["celltype"])
                    .set_index("celltype")
                )
                dy = (
                    dfm[dfm["cell_type"] == ct_y][["celltype", "mean_sampling", error_col]]
                    .dropna(subset=["celltype"])
                    .set_index("celltype")
                )

                # Align on common parts
                parts = dx.index.intersection(dy.index)
                if len(parts) == 0:
                    ax.axis("off")
                    ax.set_title(f"{m}: {ct_x} vs {ct_y}\n(no common parts)")
                    continue

                x_mean = dx.loc[parts, "mean_sampling"].to_numpy(dtype=float)
                y_mean = dy.loc[parts, "mean_sampling"].to_numpy(dtype=float)
                x_err = dx.loc[parts, error_col].to_numpy(dtype=float)
                y_err = dy.loc[parts, error_col].to_numpy(dtype=float)

                # Scatter with error bars
                ax.errorbar(x_mean, y_mean, xerr=x_err, yerr=y_err, fmt='o',
                            ecolor='gray', elinewidth=1.0, capsize=3, capthick=1.0, color='C0')

                # Optional annotations
                if annotate:
                    for xi, yi, label in zip(x_mean, y_mean, parts):
                        ax.annotate(str(label), (xi, yi), textcoords="offset points", xytext=(5, 2), fontsize=8)

                # Labels and title
                ax.set_xlabel(f"{m} ({ct_x})")
                ax.set_ylabel(f"{m} ({ct_y})")
                ax.set_title(f"{m}: {ct_x} vs {ct_y}", fontsize=10)

                # y = x reference line
                if refline:
                    lo = float(np.nanmin([x_mean.min(), y_mean.min()]))
                    hi = float(np.nanmax([x_mean.max(), y_mean.max()]))
                    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)

                ax.grid(True, alpha=0.2)

        # Hide any unused axes (if ncols > len(pairs))
        total_plots = nrows * ncols
        used_plots = nrows * len(pairs)
        for k in range(used_plots, total_plots):
            r, c = divmod(k, ncols)
            axes[r][c].axis("off")

        fig.tight_layout()
        return fig, axes
