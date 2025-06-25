import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

def generate_original_dataset(seed=42, num_samples=500, filename="dataset.csv"):
    """Generates original dataset"""
    np.random.seed(seed)
    df = pd.DataFrame({
        "Category1": np.random.choice(
            ["A", "B", "C", "D", "E"], size=num_samples, p=[0.2, 0.4, 0.2, 0.1, 0.1]
        ),
        "Value1": np.random.normal(10, 2, size=num_samples),  
        "Value2": np.random.normal(20, 6, size=num_samples),  
    })
    df.to_csv(filename, sep=";", index=False)
    return df

def generate_simplified_dataset(df, n_samples=2000, output_file="simplified_generated_dataset.csv"):
    """Generates new dataset which is similar to the original dataset"""
    mean_v1, std_v1 = df["Value1"].mean(), df["Value1"].std()
    mean_v2, std_v2 = df["Value2"].mean(), df["Value2"].std()
    cat_probs = df["Category1"].value_counts(normalize=True)

    new_df = pd.DataFrame({
        "Category1": np.random.choice(cat_probs.index, size=n_samples, p=cat_probs.values),
        "Value1": np.random.normal(mean_v1, std_v1, size=n_samples),
        "Value2": np.random.normal(mean_v2, std_v2, size=n_samples),
    })

    new_df.to_csv(output_file, index=False)
    return new_df

def perform_ks_tests(original_df, generated_df):
    """Performs statistical tests(Kolmogorov-Smirnov (KS) tests) to determine the similarity between two samples"""
    ks_value1 = ks_2samp(original_df["Value1"], generated_df["Value1"])
    ks_value2 = ks_2samp(original_df["Value2"], generated_df["Value2"])

    print(f"KS Test for Value1: statistic={ks_value1.statistic:.4f}")
    print(f"KS Test for Value2: statistic={ks_value2.statistic:.4f}")

def plot_distributions(original_df, generated_df):
    sns.kdeplot(original_df["Value1"], label="Original Value1", fill=True)
    sns.kdeplot(generated_df["Value1"], label="Generated Value1", fill=True)
    plt.title("Distribution Comparison: Value1")
    plt.legend()
    plt.show()

    sns.kdeplot(original_df["Value2"], label="Original Value2", fill=True)
    sns.kdeplot(generated_df["Value2"], label="Generated Value2", fill=True)
    plt.title("Distribution Comparison: Value2")
    plt.legend()
    plt.show()

    sns.countplot(x="Category1", data=original_df, color='blue', alpha=0.6, label='Original')
    sns.countplot(x="Category1", data=generated_df, color='red', alpha=0.4, label='Generated')
    plt.title("Category1 Distribution Comparison")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    original_df = generate_original_dataset()
    original_df = pd.read_csv("dataset.csv", sep=';')

    generated_df = generate_simplified_dataset(original_df)

    perform_ks_tests(original_df, generated_df)
    plot_distributions(original_df, generated_df)
