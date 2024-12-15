from common_imports import pd, os

def save_results_to_csv(results, experiment_name, results_dir):
    """
    Save experiment results to a CSV file.

    Args:
        results (list or dict): Experiment results to save. Should be compatible with `pandas.DataFrame`.
        experiment_name (str): Name of the experiment, used for naming the CSV file.
        results_dir (str): Directory where the CSV file will be saved.

    Saves:
        - A CSV file containing the experiment results in the specified directory.

    Prints:
        - Path to the saved CSV file.
    """
    csv_path = os.path.join(results_dir, f"{experiment_name}_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

def load_results(experiment_name, results_dir):
    """
    Load experiment results from a CSV file.

    Args:
        experiment_name (str): Name of the experiment, used for identifying the CSV file.
        results_dir (str): Directory where the CSV file is located.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded experiment results.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
    """
    csv_path = os.path.join(results_dir, f"{experiment_name}_results.csv")
    return pd.read_csv(csv_path)


