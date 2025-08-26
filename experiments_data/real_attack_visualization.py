import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_data(directory):
    """
    Parses all 'output_*.txt' files in a given directory to extract key metrics.

    Args:
        directory (str): The path to the directory containing the data files.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed data with columns for
                      measurements, incorrect_coefficients, and accuracy.
    """
    data = []
    print(f"Searching for 'output_*.txt' files in: {directory}")

    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return pd.DataFrame()  # Return an empty DataFrame

    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.startswith("output_") and filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as f:
                    lines = f.read()

                    # Use regex to find the required values. This is more robust
                    # than line-by-line reading if the file format varies slightly.
                    measurements_match = re.search(
                        r"Real number of measurements used = (\d+)", lines
                    )
                    incorrect_coeffs_match = re.search(
                        r"Average number of incorrect coefficients = ([\d\.]+)", lines
                    )
                    accuracy_match = re.search(r"empirical accuracy ([\d\.]+)", lines)
                    logical_oracle_match = re.search(
                        r"Average oracle calls used = (\d+)\.0", lines
                    )

                    # Ensure all three metrics were found before adding the data point
                    if (
                        measurements_match
                        and incorrect_coeffs_match
                        and accuracy_match
                        and logical_oracle_match
                    ):
                        measurements = int(measurements_match.group(1))
                        incorrect_coefficients = float(incorrect_coeffs_match.group(1))
                        accuracy = float(accuracy_match.group(1))
                        logical_oracle = float(logical_oracle_match.group(1))
                        data.append(
                            [
                                measurements,
                                incorrect_coefficients,
                                accuracy,
                                logical_oracle,
                            ]
                        )
                    else:
                        print(
                            f"Warning: Could not parse all required data from {filename}. Skipping."
                        )
            except Exception as e:
                print(f"Error reading or parsing file {filename}: {e}")

    if not data:
        print("No valid data was extracted. Please check the files and directory.")
        return pd.DataFrame()

    print(f"Successfully parsed {len(data)} files.")
    return pd.DataFrame(
        data,
        columns=[
            "measurements",
            "incorrect_coefficients",
            "accuracy",
            "logical_oracle",
        ],
    )


def create_plot(
    df,
    coeff_limit=20,
    measurements_limit=12000,
    accuracy_threshold=0.95,
    output_filename="real_attack_visualization.pdf",
):
    """
    Generates and saves a scatter plot from the parsed data, using different markers
    based on an accuracy threshold.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        coeff_limit (int): The upper limit for the incorrect coefficients y-axis.
        accuracy_threshold (float): The accuracy value to separate markers.
        output_filename (str): The name of the file to save the plot to.
    """
    if df.empty:
        print("Cannot create plot from empty DataFrame.")
        return

    # --- Handle the limit requirement for incorrect coefficients ---
    df["plot_incorrect_coeffs"] = df["incorrect_coefficients"].clip(upper=coeff_limit)
    df["plot_measurements"] = df["measurements"].clip(upper=measurements_limit)

    # --- Split the data based on the accuracy threshold ---
    df_high_accuracy = df[df["accuracy"] >= accuracy_threshold]
    df_low_accuracy = df[df["accuracy"] < accuracy_threshold]

    # --- Create the scatter plot ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))

    # --- Plot the two data groups with different markers ---
    # Ensure a consistent color scale by setting vmin and vmax
    vmin = df["accuracy"].min()
    vmax = df["accuracy"].max()

    # Plot low accuracy points as 'X'
    ax.scatter(
        df_low_accuracy["plot_measurements"],
        df_low_accuracy["plot_incorrect_coeffs"],
        c=df_low_accuracy["accuracy"],
        cmap="viridis",
        marker="X",
        s=180,
        alpha=0.8,
        label=f"Accuracy < {accuracy_threshold}",
        vmin=vmin,
        vmax=vmax,
    )

    # Plot high accuracy points as circles 'o'
    scatter = ax.scatter(
        df_high_accuracy["plot_measurements"],
        df_high_accuracy["plot_incorrect_coeffs"],
        c=df_high_accuracy["accuracy"],
        cmap="viridis",
        marker="o",
        s=200,
        alpha=0.8,
        edgecolors="k",
        linewidth=0.5,
        label=f"Accuracy >= {accuracy_threshold}",
        vmin=vmin,
        vmax=vmax,
    )

    # --- Customize the plot for clarity ---
    # Add a color bar to show the accuracy scale
    cbar = plt.colorbar(scatter)
    cbar.set_label("Empirical Accuracy", fontsize=26, weight="bold")
    cbar.ax.tick_params(labelsize=22)

    # Add a legend to explain the markers
    ax.legend(
        fontsize=26,
        loc="center right",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Set labels and title
    ax.set_xlabel("Number of Prime+Probe Measurements", fontsize=26, weight="bold")
    ax.set_ylabel("Hamming Distance", fontsize=26, weight="bold")
    # ax.set_title("Secret Key Recovery Performance", fontsize=18, weight="bold")

    # # --- Customize Y-axis for the limit label ---
    # ax.set_ylim(-coeff_limit * 0.05, coeff_limit * 1.05)

    # # Generate evenly spaced ticks. A step of 5 is nice for a limit of 20.
    # tick_step = 2
    # new_ticks = np.arange(0, coeff_limit + 1, tick_step)

    # # Create the labels for our new ticks
    # tick_labels = [f"{int(t)}" for t in new_ticks]

    # # Change the label for the top tick to indicate 'and above'
    # if tick_labels:
    #     tick_labels[-1] = f"{int(new_ticks[-1])}+"

    # ax.set_yticks(new_ticks)
    # ax.set_yticklabels(tick_labels)

    # --- Customize Y-axis ---
    ax.set_ylim(-coeff_limit * 0.05, coeff_limit * 1.05)
    tick_step_y = 2
    new_ticks_y = np.arange(0, coeff_limit + 1, tick_step_y)
    tick_labels_y = [f"{int(t)}" for t in new_ticks_y]
    if tick_labels_y:
        tick_labels_y[-1] = f"{int(new_ticks_y[-1])}+"
    ax.set_yticks(new_ticks_y)
    ax.set_yticklabels(tick_labels_y)

    # --- Customize X-axis ---
    ax.set_xlim(4000, measurements_limit * 1.05)
    tick_step_x = 2000
    new_ticks_x = np.arange(4000, measurements_limit + 1, tick_step_x)
    # Use 'k' for thousands to keep labels clean
    tick_labels_x = [f"{int(t)}" for t in new_ticks_x]
    if tick_labels_x:
        tick_labels_x[-1] = f"{int(new_ticks_x[-1])}+"
    ax.set_xticks(new_ticks_x)
    ax.set_xticklabels(tick_labels_x)

    # Improve overall aesthetics
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # fig.tight_layout()

    # Save the plot to a file
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to '{output_filename}'")


if __name__ == "__main__":
    # --- Set up Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Parse timing side-channel attack data and generate a visualization.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory containing the 'output_{i}.txt' files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="The upper limit for the 'incorrect coefficients' y-axis. (default: 20)",
    )
    parser.add_argument(
        "--x_limit",
        type=int,
        default=12000,
        help="The upper limit for the 'measurements' x-axis. (default: 12000)",
    )
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=0.95,
        help="Accuracy threshold to distinguish markers (default: 0.95)",
    )

    args = parser.parse_args()

    # Parse the data from the specified directory
    data_df = parse_data(args.directory)

    if not data_df.empty:
        well_behaved_df = data_df[data_df["incorrect_coefficients"] <= 4]
        num_well_behaved = len(well_behaved_df)

        print("\n--- Analysis of Well-Behaved Cases (Incorrect Coefficients <= 4) ---")
        avg_accuracy = well_behaved_df["accuracy"].mean()
        avg_measurements = well_behaved_df["measurements"].mean()
        avg_coeffs = well_behaved_df["incorrect_coefficients"].mean()
        avg_log_oracles = well_behaved_df["logical_oracle"].mean()

        print(f"Number of cases: {num_well_behaved}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Number of Measurements: {avg_measurements:.2f}")
        print(f"Average Number of Incorrect Coefficients: {avg_coeffs:.2f}")
        print(
            f"Average Number of Repetitions: {(avg_measurements / avg_log_oracles):.2f}"
        )

        print("-----------------------------------------------------------------\n")
        print("--- Analysis of All Cases ---")
        print(f"Average Accuracy: {data_df['accuracy'].mean():.4f}")
        print(f"Average Number of Measurements: {data_df['measurements'].mean():.2f}")
        print(
            f"Average Number of Incorrect Coefficients: {data_df['incorrect_coefficients'].mean():.2f}"
        )

    # Create and save the plot
    if not data_df.empty:
        create_plot(
            data_df,
            coeff_limit=args.limit,
            measurements_limit=args.x_limit,
            accuracy_threshold=args.accuracy_threshold,
        )
