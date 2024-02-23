import argparse
import logging
from pathlib import Path
import time
import apache_beam as beam
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Iterable

logging.basicConfig(level=logging.INFO)


class TokenizeText(beam.DoFn):
    """
    Apache Beam DoFn for tokenizing text using a given tokenizer.
    """

    def __init__(self, tokenizer):
        """
        Initializes the TokenizeText instance.

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to be used for tokenization.
        """
        self.tokenizer = tokenizer

    def process(self, text: str) -> Iterable[int]:
        """
        Tokenizes the input text and yields the length of tokens.

        Args:
            text (str): Input text to be tokenized.

        Yields:
            int: Length of tokens obtained after tokenization.
        """
        tokens = self.tokenizer.encode(text)
        yield len(tokens)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset",
        default="anirudhlakhotia/roots_indic-hi_iitb_english_hindi_corpus",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to the tokenizer",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
        default="output",
    )

    return parser.parse_args()


def process_token_counts(total_token_count: int):
    """
    Processes the total token count obtained.

    Args:
        total_token_count (int): Total count of tokens.
    """
    print(f"Total tokens: {total_token_count}")


def main(args: argparse.Namespace):
    """
    Main function for executing the tokenization pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    try:
        dataset = load_dataset(args.dataset_name, split="train")
        dataset = dataset.shuffle(seed=42).select(range(100))
        print("Dataset loaded")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

        with beam.Pipeline() as pipeline:
            texts = pipeline | "CreateTexts" >> beam.Create(dataset["text"])

            total_token_count = (
                texts
                | "Tokenize" >> beam.ParDo(TokenizeText(tokenizer))
                | "CombineCounts" >> beam.CombineGlobally(sum)
            )

            total_token_count | "ProcessCounts" >> beam.Map(process_token_counts)

    except Exception as e:
        logging.error("Error: %s", e)


if __name__ == "__main__":
    time_start = time.perf_counter()
    args = parse_args()
    main(args)
    time_end = time.perf_counter()

    print(f"Time taken: {time_end - time_start:.2f} seconds")
