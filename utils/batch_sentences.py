import argparse
import re
from typing import List, Tuple
from datasets import Dataset, load_dataset


def batch_sentences(examples: Dataset, max_words: int = 200) -> List[str]:
    """
    Batches sentences from a dataset into chunks of a specified maximum number of words.

    Args:
        examples (Dataset): The dataset containing the sentences.
        max_words (int, optional): The maximum number of words per batch. Defaults to 200.

    Returns:
        List[str]: A list of batched sentences, where each sentence is a string.
    """
    batched_data = []
    current_batch = ""
    for example in examples["text"]:
        sentence = example
        sentence = re.sub(r"</s>", "", sentence)  # Remove </s> tokens
        words = sentence.strip().split()
        for word in words:
            if len(current_batch.split()) + 1 <= max_words:
                current_batch += word + " "
            else:
                # Add </s> token at the end of each batched row
                batched_data.append(current_batch.strip() + "</s>")
                current_batch = word + " "
    if current_batch:  # Add the last batch if any remaining
        batched_data.append(current_batch.strip() + "</s>")
    return batched_data


def main(
    dataset_name: str, split: str, max_words: int, num_proc: int
) -> Tuple[Dataset, Dataset]:
    """
    Loads a dataset, batches the sentences, and returns the original and batched datasets.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (e.g., "train", "test", "validation").
        max_words (int): The maximum number of words per batch.
        num_proc (int): The number of processes to use for batching.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the original dataset and the batched dataset.
    """
    dataset = load_dataset(dataset_name, split=split)

    def batch_sentences_wrapper(examples):
        return {"text": batch_sentences(examples, max_words)}

    batched_dataset = dataset.map(
        batch_sentences_wrapper, batched=True, batch_size=128000, num_proc=num_proc
    )
    return dataset, batched_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch sentences in a dataset")
    parser.add_argument(
        "dataset_name", type=str, help="The name of the dataset to load"
    )
    parser.add_argument(
        "split",
        type=str,
        help='The split of the dataset to load (e.g., "train", "test", "validation")',
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=200,
        help="The maximum number of words per batch (default: 200)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=24,
        help="The number of processes to use for batching (default: 24)",
    )

    args = parser.parse_args()

    dataset, batched_dataset = main(
        args.dataset_name, args.split, args.max_words, args.num_proc
    )
    print(f"Original dataset: {dataset}")
    print(f"Batched dataset: {batched_dataset}")
