
# Import the required modules
import nest_asyncio
import matplotlib.pyplot as plt
import pandas as pd
from galactic import GalacticDataset

#  Apply the nest_asyncio module to enable async coroutines in a notebook
# nest_asyncio.apply()

# Define a constant for the Hugging Face dataset name
DATASET_NAME = "anirudhlakhotia/Hindi-PreTrainingData"

# Define a function to load the dataset from Hugging Face and count the tokens
def load_and_count_tokens(split: str, fields: list, tokenizer: str) -> dict:
    """
    Load the dataset from Hugging Face and count the tokens in the specified fields.

    Parameters:
    split (str): The split of the dataset to load, such as "train" or "test".
    fields (list): The list of fields to count the tokens in, such as ["text"].
    tokenizer (str): The name of the tokenizer to use, such as "AsphyXIA/baarat-hin-en-0.1".

    Returns:
    dict: A dictionary with the field names as keys and the token counts as values.
    """
    # Load the dataset from Hugging Face
    ds = GalacticDataset.from_hugging_face(DATASET_NAME, split=split)
    # Count the tokens in the specified fields using the tokenizer
    dataset = ds.count_tokens(fields=fields, tokenizer=tokenizer)
    # Return the dictionary with the token counts
    return dataset

# Define a function to plot the histogram of the token counts
def plot_token_histogram(token_counts: list, bins: int, range: tuple) -> None:
    """
    Plot the histogram of the token counts using matplotlib.

    Parameters:
    token_counts (list): The list of token counts to plot, such as dataset["__token_count__text"].
    bins (int): The number of bins to use in the histogram, such as 100.
    range (tuple): The range of values to plot in the histogram, such as (0, 513).

    Returns:
    None: This function does not return anything, but it shows the plot using plt.show().
    """
    # Plot the histogram of the token counts using matplotlib
    plt.hist(token_counts, bins=bins, range=range)
    # Add labels and title to the plot
    plt.xlabel("Token count")
    plt.ylabel("Frequency")
    plt.title("Histogram of token counts in the text field")
    # Show the plot
    plt.show()

# Define a function to count the bins of the token counts
def count_token_bins(token_counts: list, bins: list) -> pd.Series:
    """
    Count the bins of the token counts using pandas.

    Parameters:
    token_counts (list): The list of token counts to count, such as dataset["__token_count__text"].
    bins (list): The list of bin edges to use, such as [0, 100, 200, 300, 400, 500, 600].

    Returns:
    pd.Series: A pandas Series with the bin labels as the index and the bin counts as the values.
    """
    # Convert the list of token counts to a pandas DataFrame
    token_counts_df = pd.DataFrame(token_counts)
    # Use the value_counts() method on the DataFrame with the bins argument
    bin_counts = token_counts_df[0].value_counts(bins=bins)
    # Return the bin counts as a pandas Series
    return bin_counts



if __name__ == "__main__":
    # Load the dataset from Hugging Face and count the tokens in the text field
    dataset = load_and_count_tokens(split="train", fields=["text"], tokenizer="AsphyXIA/baarat-hin-en-0.1")

    # Plot the histogram of the token counts in the text field
    plot_token_histogram(token_counts=dataset["__token_count__text"], bins=100, range=(0, 513))

    # Count the bins of the token counts in the text field
    bin_counts = count_token_bins(token_counts=dataset["__token_count__text"], bins=[0, 100, 200, 300, 400, 500, 600])

    # Print the bin counts
    print(bin_counts)