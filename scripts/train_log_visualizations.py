import os
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.Trainer import TrainingRunIO
from src.Util.Paths import RUNS_DIR

COLOR_PALETTE = ["D4A5A5", "FFB3BA", "FFDFBA", "BAE1FF", "B5E7A0", "C3B1E1", "BAFFC9", "FF91BA"]
# convert to color values
COLOR_PALETTE = [tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)) for color in COLOR_PALETTE]


# Set default font size for various elements
def set_medium_font_size():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )


def set_large_font_size():
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.titlesize": 22,
        }
    )


set_medium_font_size()

LINE_WIDTH = 2


def add_event_id_padding(
    df: pd.DataFrame,
    event_key: str = TrainingRunIO.EVENT_KEY,
    no_event_key: str = TrainingRunIO.NO_EVENT,
    shift: bool = True,
) -> pd.DataFrame:
    """
    Add padding to event IDs if the batch is not full.

    :param df pd.Dataframe: Dataframe containing the event IDs
    :param event_key str: Special string only found in event_id columns
    :param no_event_key str: Special string for empty event IDs
    :param shift bool: Whether to shift the event IDs or fill with no_event_key
    :return pd.Dataframe: Dataframe with padded event IDs
    """

    event_id_cols = [col for col in df.columns if event_key in col]
    event_cols_start = df.columns.get_loc(event_id_cols[0])
    batch_size = len(event_id_cols)

    for i in range(len(df)):
        num_event_ids = len([val for val in df.iloc[i] if isinstance(val, str) and event_key in val])
        if num_event_ids < batch_size:
            shift_amount = batch_size - num_event_ids
            shift_start = event_cols_start + num_event_ids
            if shift:
                df.iloc[i, shift_start:] = df.iloc[i, shift_start:].shift(shift_amount, fill_value=no_event_key)
            else:
                df.iloc[i, shift_start : shift_start + shift_amount] = no_event_key

    return df


def clamp_outliers(
    log: pd.DataFrame, field: str, num_std: int = 5, minimum: float = 0.0, max_quantile=0.98
) -> pd.Series:
    """
    Clamp the outliers in a certain field of the log to a mean +/- multiple of std

    :param log pd.DataFrame: Dataframe containing the loss values
    :param field str: The field to clamp the outliers
    :param num_std int: The number of standard deviations to clamp the outliers
    :return pd.DataFrame: Dataframe with clamped outliers
    """

    filtered = log[log[field].notna()][field]
    filtered = filtered[~filtered.isin([float("inf"), float("-inf")])]
    max_valid = filtered.max()

    # Clamp outliers that result from numerical instability or other issues
    filtered = log[field].fillna(max_valid)
    filtered = filtered.replace([float("inf"), float("-inf")], max_valid)
    filtered = filtered.apply(lambda x: max(minimum, x))

    # Clamp to quantile
    max_val = filtered.quantile(max_quantile)
    filtered = filtered.apply(lambda x: min(max_val, x))

    mean = filtered.mean()
    std = filtered.std()
    return filtered.apply(lambda x: min(mean + num_std * std, max(mean - num_std * std, x)))


def plot_loss(log: pd.DataFrame, figures_dir: str):
    """
    Plots the loss curve for a certain log of runtime information.

    :param log pd.Dataframe: Dataframe containing the loss values
    :param figures_dir str: The directory of the run figures
    """
    epoch_field = TrainingRunIO.EPOCH_FIELD
    loss_field = TrainingRunIO.LOSS_FIELD
    loss_log = log.loc[:, [epoch_field, loss_field]]

    set_large_font_size()
    plt.figure(figsize=(10, 6))
    plt.plot(loss_log[epoch_field], loss_log[loss_field], marker="o", color=COLOR_PALETTE[1], linewidth=LINE_WIDTH)
    # plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "loss.png"), dpi=300)
    plt.close()
    set_medium_font_size()


def plot_shell_metrics(log: pd.DataFrame, figures_dir: str, metric: str, metric_text: str, use_log: bool = True):
    """
    Plots the metric curves for both the train and eval values of the eval.log per epoch

    :param log pd.Dataframe: Dataframe containing the loss values
    :param figures_dir str: The directory of the run figures
    :param metric str: The metric to plot
    :param metric_text str: The text to display for the metric
    :param use_log bool: Whether to use the log of the metric
    """
    nr_shells = len([col for col in log.columns if "mse_train_" in col])
    epoch_field = TrainingRunIO.EPOCH_FIELD
    num_epochs = log[epoch_field].max() + 1
    metric_u = metric.upper()

    def create_fig(save_path: str):
        plt.xlabel("Epochs")
        plt.ylabel(f"Log {metric_text}" if use_log else metric_text)
        # plt.title(f"{metric_u} per Shell Across Epochs {skip_start} to {num_epochs}")
        plt.legend(title="Shells")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    for skip_start in [0, num_epochs // 10]:
        dataset_keys = ["train", "val"]
        dataset_avgs = {key: [] for key in dataset_keys}
        for value in dataset_keys:
            set_large_font_size()
            plt.figure(figsize=(10, 6))

            for i in range(1, nr_shells + 1):
                metric_field = f"{metric}_{value}_{i}"
                metric_log = log.loc[skip_start:, [epoch_field, metric_field]]
                metric_log[metric_field] = clamp_outliers(metric_log, metric_field, 3)

                values = np.log10(metric_log[metric_field]) if use_log else metric_log[metric_field]
                plt.plot(metric_log[epoch_field], values, label=i, color=COLOR_PALETTE[i], linewidth=LINE_WIDTH)

                if len(dataset_avgs[value]) == 0:
                    dataset_avgs[value] = metric_log[metric_field]
                else:
                    dataset_avgs[value] += metric_log[metric_field]
            dataset_avgs[value] /= nr_shells - 1
            plt.xticks(range(skip_start, num_epochs, num_epochs // 10))

            create_fig(os.path.join(figures_dir, f"{metric_u}_per_shell_{value}_from_{skip_start}.png"))

        set_medium_font_size()
        values = np.log10(dataset_avgs["train"]) if use_log else dataset_avgs["train"]
        plt.plot(
            values,
            label=f"Train Avg {metric_u}",
            color=COLOR_PALETTE[2],
            linestyle="-",
            marker="o",
            linewidth=LINE_WIDTH,
        )
        values = np.log10(dataset_avgs["val"]) if use_log else dataset_avgs["val"]
        plt.plot(
            values,
            label=f"Val Avg {metric_u}",
            color=COLOR_PALETTE[3],
            linestyle="--",
            marker="s",
            linewidth=LINE_WIDTH,
        )
        plt.xticks(range(skip_start, num_epochs, num_epochs // 10))
        create_fig(os.path.join(figures_dir, f"{metric_u}_from_{skip_start}.png"))


def plot_mse_shells(log: pd.DataFrame, figures_dir: str, use_log: bool):
    """
    Plots the MSE curves for both the train and eval values of the eval.log per epoch

    :param log pd.Dataframe: Dataframe containing the loss values
    :param figures_dir str: The directory of the run figures
    :param use_log bool: Whether to use the log of the metric
    """
    plot_shell_metrics(
        log=log, figures_dir=figures_dir, metric="mse", metric_text="Mean Squared Error (MSE)", use_log=use_log
    )


def plot_hd_shells(log: pd.DataFrame, figures_dir: str, use_log: bool):
    """
    Plots the HD curves for both the train and eval values of the eval.log per epoch

    :param log pd.Dataframe: Dataframe containing the loss values
    :param figures_dir str: The directory of the run figures
    :param use_log bool: Whether to use the log of the metric
    """
    plot_shell_metrics(
        log=log, figures_dir=figures_dir, metric="hd", metric_text="Hausdorff Distance (HD)", use_log=use_log
    )


def get_min_loss_metrics(log: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Get the minimum loss metrics from the log file.

    :param log pd.DataFrame: Dataframe containing the loss values
    :return tuple[list[float], list[float]]: Tuple of lists containing the min loss MSE and HD
    """
    min_loss_epoch = log.loc[log[TrainingRunIO.LOSS_FIELD].idxmin()][TrainingRunIO.EPOCH_FIELD]
    mse_fields = [col for col in log.columns if "mse_val_" in col]
    hd_fields = [col for col in log.columns if "hd_val_" in col]

    return (log.loc[min_loss_epoch, mse_fields].values.mean(), log.loc[min_loss_epoch, hd_fields].values.mean())


def create_metrics_table(mses: list[float], hds: list[float], names: list[str], out_dir: str = RUNS_DIR):
    """
    Create a table of the MSE and HD metrics for each run.

    :param mses list[float]: List of MSE metrics
    :param hds list[float]: List of HD metrics
    :param out_dir str: Output directory for the table
    """
    data = {names[i]: [mses[i], hds[i]] for i in range(len(mses))}
    df = pd.DataFrame(data, index=["MSE", "HD"])

    with open(os.path.join(out_dir, "metrics_table.tex"), "w") as f:
        f.write(df.to_latex())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--runs_dir", default=RUNS_DIR, help="Directory containing the runs")
    ap.add_argument("-i", "--run_ids", nargs="+", help="Run IDs to visualize")
    ap.add_argument("--names", default=[], nargs="+", help="Names of the runs")
    ap.add_argument("-m", "--modify", action="store_true", help="Modify the log files")
    ap.add_argument("--make_table", action="store_true", help="Whether to make a table of the metrics")
    ap.add_argument("--use_log", action="store_true", help="Whether to use the log of the metric")
    args = ap.parse_args()

    min_loss_mses = []
    min_loss_hds = []

    for run_id in args.run_ids:
        run_dir = os.path.join(args.runs_dir, run_id)
        eval_log = pd.read_csv(os.path.join(run_dir, TrainingRunIO.EVAL_LOG_FILE))

        # Shift rows for missing batch entries
        if args.modify:
            train_log = pd.read_csv(os.path.join(run_dir, TrainingRunIO.TRAIN_LOG_FILE))
            train_log = add_event_id_padding(df=train_log, shift=False)
            train_log.to_csv(os.path.join(run_dir, TrainingRunIO.TRAIN_LOG_FILE))

        figures_dir = os.path.join(run_dir, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        plot_loss(log=eval_log, figures_dir=figures_dir)
        plot_mse_shells(log=eval_log, figures_dir=figures_dir, use_log=args.use_log)
        plot_hd_shells(log=eval_log, figures_dir=figures_dir, use_log=args.use_log)

        min_loss_mse, min_loss_hd = get_min_loss_metrics(eval_log)
        min_loss_mses.append(min_loss_mse)
        min_loss_hds.append(min_loss_hd)

    if args.make_table:
        print(f"Min loss MSE: {min_loss_mses}")
        print(f"Lowest MSE: {min(min_loss_mses)}, run id: {args.run_ids[min_loss_mses.index(min(min_loss_mses))]}")

        print(f"Min loss HD: {min_loss_hds}")
        print(f"Lowest HD: {min(min_loss_hds)}, run id: {args.run_ids[min_loss_hds.index(min(min_loss_hds))]}")

        nr_names = len(args.names)
        nr_runs = len(args.run_ids)
        if nr_names < nr_runs:
            args.names.extend(args.run_ids[nr_names:])
        create_metrics_table(min_loss_mses, min_loss_hds, args.names)
