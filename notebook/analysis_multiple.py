# %% [markdown]
# # Multi-Plot Analysis & Metrics Summary
# This script generates multiple plots (one per metric type) and produces a comprehensive summary table comparing all experiments.

# %%
from src.view import fetch_runs, plot, plot_with_var
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

path = "note/fonts/HKGrotesk-Regular.ttf"
prop = font_manager.FontProperties(fname=path)

# %% [markdown]
# ## Fetch Data

# %%
experiments, _, _, _ = fetch_runs(skip_n=50)

# %% [markdown]
# ## Summary Table: Final Metrics Comparison

# %%
def compute_final_metrics(exp):
    """Extract final metric values from each run."""
    metrics_df = exp["metrics"]
    
    result = {
        "run": exp["name"],
        "test_acc": exp.get("test_acc", None),
    }
    
    # Get last non-null value for each metric
    for col in metrics_df.columns:
        if col != "epoch" and col != "val/acc":
            values = metrics_df[col].dropna()
            if len(values) > 0:
                result[col] = values.iloc[-1]
    
    return result

summary_df = pd.DataFrame([compute_final_metrics(exp) for exp in experiments])
print(summary_df.to_string())

# %% [markdown]
# ## Plot 1: Validation Loss vs Accuracy

# %%
plot(
    data={exp["name"]: exp["metrics"][[("epoch", "val/loss"), ("epoch", "val/acc")]] 
          for exp in experiments[:10]},
    exp_name="Validation Loss & Accuracy",
    x_label="Epochs",
    y_label="Metric",
    fig_name="val_loss_acc",
    skip_n=50,
)

# %% [markdown]
# ## Plot 2: Compression Rates

# %%
def plot_metrics(metrics_df, metric_col, exp_list, plot_type="line"):
    """Plot a single metric column across multiple runs."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ["red", "blue", "green", "orange", "pink", "teal", "coral", "lavender", "purple", "gold", "lime", "navy", "crimson", "turquoise", "maroon", "olive", "indigo", "salmon", "sienna", "orchid", "khaki", "steelblue", "darkseagreen"]
    
    for i, exp in enumerate(exp_list[:20]):  # Limit to 20 runs
        name = exp["name"]
        metrics = exp["metrics"]
        
        # Find the epoch and metric columns
        epoch_col = None
        metric_col_exp = None
        
        for col in metrics.columns:
            if col == "epoch":
                epoch_col = col
            elif col == metric_col:
                metric_col_exp = col
        
        if epoch_col is None or metric_col_exp is None:
            continue
            
        epochs = metrics[epoch_col].dropna()
        values = metrics[metric_col_exp].dropna()
        
        if len(epochs) > 0 and len(values) > 0:
            ax.plot(epochs, values, marker='o', markersize=5, label=name, 
                    color=colors[i % len(colors)], linewidth=2)
    
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel("Epochs", fontproperties=prop, fontsize=13)
    ax.set_ylabel(metric_col, fontproperties=prop, fontsize=13)
    ax.ticklabel_format(axis="both", useMathText=True, useOffset=True, style="sci", scilimits=(0, 0))
    ax.set_title(f"{metric_col} Across Runs", fontproperties=prop, fontsize=14)
    ax.grid(True)
    ax.set_xlim(left=0)
    
    if metric_col in ["gzip_compression_rate", "bz2_compression_rate", "lzma_compression_rate", "sparsity", "weight_l2"]:
        ax.set_ylim(bottom=0)
    
    if fig_name := getattr(plot_metrics, 'fig_name', None):
        fig.savefig(str(fig_name) + "_" + metric_col + ".pdf")
        print(f"Saved: {fig_name}_{metric_col}.pdf")
    
    return fig, ax

# %% [markdown]
# ### Gzip Compression Rate

# %%
plot_metrics.fig_name = "gzip_summary"
plot_metrics("gzip_compression_rate", experiments, "gzip_summary")

# %% [markdown]
# ### BZ2 Compression Rate

# %%
plot_metrics.fig_name = "bz2_summary"
plot_metrics("bz2_compression_rate", experiments, "bz2_summary")

# %% [markdown]
# ### LZMA Compression Rate

# %%
plot_metrics.fig_name = "lzma_summary"
plot_metrics("lzma_compression_rate", experiments, "lzma_summary")

# %% [markdown]
# ### Sparsity

# %%
plot_metrics.fig_name = "sparsity_summary"
plot_metrics("sparsity", experiments, "sparsity_summary")

# %% [markdown]
# ### L2 Norm

# %%
plot_metrics.fig_name = "l2_summary"
plot_metrics("weight_l2", experiments, "l2_summary")

# %% [markdown]
# ## Plot 3: BDM Complexity (Final Values)

# %%
bdm_values = summary_df["bdm_complexity"].dropna()
if len(bdm_values) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(bdm_values.index.astype(str), bdm_values.values, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Run", fontproperties=prop, fontsize=13)
    ax.set_ylabel("BDM Complexity", fontproperties=prop, fontsize=13)
    ax.ticklabel_format(axis="y", useMathText=True, style="sci", scilimits=(0, 0))
    ax.set_title("BDM Complexity - Final Values", fontproperties=prop, fontsize=14)
    ax.grid(True, axis='y')
    fig.savefig("bdm_complexity_bar.pdf")
    print("Saved: bdm_complexity_bar.pdf")

# %% [markdown]
# ## Plot 4: Test Accuracy (Final Values)

# %%
test_acc_values = summary_df["test_acc"].dropna()
if len(test_acc_values) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(test_acc_values.index.astype(str), test_acc_values.values, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Run", fontproperties=prop, fontsize=13)
    ax.set_ylabel("Test Accuracy", fontproperties=prop, fontsize=13)
    ax.set_title("Test Accuracy - Final Values", fontproperties=prop, fontsize=14)
    ax.grid(True, axis='y')
    fig.savefig("test_accuracy_bar.pdf")
    print("Saved: test_accuracy_bar.pdf")

# %% [markdown]
# ## Plot 5: All Quantization Metrics (Combined)

# %%
def plot_combined_metrics(experiments, fig_name):
    """Create a multi-panel plot showing all quantization metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()
    
    colors = ["red", "blue", "green", "orange", "pink", "teal", "coral", "lavender", "purple", "gold", "lime", "navy", "crimson", "turquoise", "maroon", "olive", "indigo", "salmon", "sienna", "orchid", "khaki", "steelblue", "darkseagreen"]
    
    metric_cols = [
        "bdm_complexity",
        "gzip_compression_rate", 
        "bz2_compression_rate",
        "lzma_compression_rate",
        "sparsity",
        "weight_l2",
        "quant/distortion",
        "quant/changed_weights",
    ]
    
    for idx, (ax, metric_col) in enumerate(zip(axes, metric_cols)):
        if idx >= len(axes):
            ax.axis('off')
            continue
            
        ax.set_title(metric_col, fontproperties=prop, fontsize=12)
        ax.set_xlabel("Epochs", fontproperties=prop, fontsize=11)
        ax.set_ylabel(metric_col, fontproperties=prop, fontsize=11)
        ax.grid(True)
        
        # Plot multiple runs
        for i, exp in enumerate(experiments[:15]):
            metrics = exp["metrics"]
            epoch_col = None
            metric_col_exp = None
            
            for col in metrics.columns:
                if col == "epoch":
                    epoch_col = col
                elif col == metric_col:
                    metric_col_exp = col
            
            if epoch_col is None or metric_col_exp is None:
                continue
                
            epochs = metrics[epoch_col].dropna()
            values = metrics[metric_col_exp].dropna()
            
            if len(epochs) > 0 and len(values) > 0:
                ax.plot(epochs, values, marker='.', markersize=4, 
                        label=exp["name"][:30], 
                        color=colors[i % len(colors)], alpha=0.6)
        
        if metric_col in ["gzip_compression_rate", "bz2_compression_rate", "lzma_compression_rate", "sparsity", "weight_l2"]:
            ax.set_ylim(bottom=0)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    fig.suptitle("Quantization Metrics - Multiple Runs Comparison", 
                 fontproperties=prop, fontsize=16, fontweight='bold')
    
    if fig_name:
        fig.savefig(str(fig_name) + "_combined.pdf", bbox_inches='tight')
        print(f"Saved: {fig_name}_combined.pdf")
    
    return fig, axes

# %% [markdown]
# ### Combined Quantization Metrics Plot

# %%
plot_combined_metrics.fig_name = "quant_metrics"
plot_combined_metrics(experiments, "quant_metrics")

# %% [markdown]
# ## Summary Statistics

# %%
print("\n=== FINAL METRICS SUMMARY ===")
print(f"\nTotal runs analyzed: {len(experiments)}")
print(f"\n{summary_df[['run', 'test_acc', 'gzip_compression_rate', 'bz2_compression_rate', 'lzma_compression_rate', 'sparsity', 'bdm_complexity']].to_string(index=False)}")

# %%
# Save summary to CSV
summary_df.to_csv("final_metrics_summary.csv", index=False)
print("\nSaved: final_metrics_summary.csv")
