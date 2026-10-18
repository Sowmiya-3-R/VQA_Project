"""
Build Vocabulary and Top Answers for VQA.

This script:
1. Reads the dataset questions (and optional annotations).
2. Builds `answers.json`: A list of the top 3000 most frequent answers (index = class ID).
3. Builds `vocab.json`: A mapping of words appearing >= 2 times to unique integer IDs,
   with reserved tokens:
       <pad>: 0
       <unk>: 1
4. Prints:
   - Total dataset size (number of questions)
   - Number of distinct answers
   - Percentage of questions whose answers fall within the top 3000.

Usage:
    python data/build_vocab.py --questions data/questions.json \
                               --vocab_out vocab.json \
                               --answers_out answers.json \
                               --min_freq 2 \
                               --num_answers 3000
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path so we can import from text_encoder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from text_encoder.tokenizer import tokenize_text, PAD_TOKEN, UNK_TOKEN, PAD_ID, UNK_ID


def extract_question_and_answer(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extracts the question string and answer string from a raw dataset item.
    Supports multiple common VQA formats:
    - Standard dict: {"question": "...", "answer": "..."}
    - Alternative keys: "question_str", "text", "multiple_choice_answer"
    - VQA v2 annotations with list of answer dicts: {"answers": [{"answer": "yes"}, ...]}
    """
    # 1. Extract question text
    question_text = (
        item.get("question")
        or item.get("question_str")
        or item.get("text")
        or ""
    )

    # 2. Extract answer text
    answer_text = ""
    if "answer" in item and isinstance(item["answer"], str):
        answer_text = item["answer"]
    elif "multiple_choice_answer" in item and isinstance(item["multiple_choice_answer"], str):
        answer_text = item["multiple_choice_answer"]
    elif "answers" in item:
        # If list of annotator answers (e.g. VQA v2 annotations)
        raw_answers = item["answers"]
        if isinstance(raw_answers, list) and len(raw_answers) > 0:
            if isinstance(raw_answers[0], dict) and "answer" in raw_answers[0]:
                # Find consensus or most common answer among annotators
                ans_list = [a.get("answer", "") for a in raw_answers if isinstance(a, dict)]
                if ans_list:
                    answer_text = Counter(ans_list).most_common(1)[0][0]
            elif isinstance(raw_answers[0], str):
                answer_text = Counter(raw_answers).most_common(1)[0][0]

    # Standardize answer: lowercased and stripped of whitespace
    return question_text.strip(), answer_text.strip().lower()


def load_dataset(
    questions_path: str,
    annotations_path: str = None
) -> List[Tuple[str, str]]:
    """
    Loads questions and paired answers from JSON file(s).

    Args:
        questions_path: Path to questions JSON.
        annotations_path: Optional path to separate annotations JSON (e.g. VQA v2).

    Returns:
        List of (question_text, answer_text) tuples.
    """
    if not os.path.isfile(questions_path):
        raise FileNotFoundError(f"Questions file not found at: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    # Handle dictionary container (e.g. {"questions": [...]})
    if isinstance(q_data, dict) and "questions" in q_data:
        q_list = q_data["questions"]
    elif isinstance(q_data, list):
        q_list = q_data
    else:
        raise ValueError("Unsupported format in questions JSON: expected a list or dict with 'questions' key.")

    # If separate annotations file is provided, merge by question_id
    if annotations_path:
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f"Annotations file not found at: {annotations_path}")
        with open(annotations_path, "r", encoding="utf-8") as f:
            a_data = json.load(f)

        if isinstance(a_data, dict) and "annotations" in a_data:
            a_list = a_data["annotations"]
        elif isinstance(a_data, list):
            a_list = a_data
        else:
            raise ValueError("Unsupported format in annotations JSON.")

        # Map question_id to annotation
        anno_map = {item["question_id"]: item for item in a_list if "question_id" in item}

        merged_pairs = []
        for q_item in q_list:
            qid = q_item.get("question_id")
            a_item = anno_map.get(qid, {})
            # Merge keys from both
            combined = {**q_item, **a_item}
            q_text, a_text = extract_question_and_answer(combined)
            merged_pairs.append((q_text, a_text))
        return merged_pairs

    # Otherwise, extract directly from each question entry
    pairs = [extract_question_and_answer(item) for item in q_list]
    return pairs


def build_vocab_and_answers(
    pairs: List[Tuple[str, str]],
    min_freq: int = 2,
    num_answers: int = 3000
) -> Tuple[Dict[str, int], List[str], Dict[str, Any]]:
    """
    Builds vocabulary dictionary and top answers list from question-answer pairs.

    Args:
        pairs: List of (question, answer) tuples.
        min_freq: Minimum frequency for a word to be included in vocab.json.
        num_answers: Number of top frequent answers to keep for answers.json.

    Returns:
        vocab: Dictionary mapping word -> integer ID.
        top_answers: List of strings (length <= num_answers), index = class ID.
        stats: Dictionary containing summary statistics.
    """
    word_counter = Counter()
    answer_counter = Counter()

    for q_text, a_text in pairs:
        # Tokenize question using whitespace split after lowercasing
        tokens = tokenize_text(q_text)
        word_counter.update(tokens)

        # Count answer if present
        if a_text:
            answer_counter[a_text] += 1

    # 1. Build Vocabulary
    # Start with reserved tokens
    vocab = {
        PAD_TOKEN: PAD_ID,  # 0
        UNK_TOKEN: UNK_ID   # 1
    }
    next_id = 2

    # Sort words by frequency (descending), then alphabetically for determinism
    for word, count in sorted(word_counter.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_freq:
            vocab[word] = next_id
            next_id += 1

    # 2. Build Top Answers List (index = class ID)
    # Sort answers by frequency (descending)
    most_common_answers = answer_counter.most_common(num_answers)
    top_answers = [ans for ans, _ in most_common_answers]

    # 3. Calculate Coverage Statistics
    dataset_size = len(pairs)
    distinct_answers = len(answer_counter)
    top_answers_set = set(top_answers)

    covered_count = sum(1 for _, a_text in pairs if a_text in top_answers_set)
    coverage_pct = (covered_count / dataset_size * 100.0) if dataset_size > 0 else 0.0

    stats = {
        "dataset_size": dataset_size,
        "distinct_answers": distinct_answers,
        "vocab_size": len(vocab),
        "num_top_answers": len(top_answers),
        "covered_questions": covered_count,
        "coverage_percentage": coverage_pct
    }

    return vocab, top_answers, stats


def main():
    parser = argparse.ArgumentParser(description="Build VQA vocabulary and top answers.")
    parser.add_argument("--questions", type=str, default="data/questions.json",
                        help="Path to the questions JSON file.")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Optional path to separate annotations JSON file.")
    parser.add_argument("--vocab_out", type=str, default="vocab.json",
                        help="Path to save output vocab.json.")
    parser.add_argument("--answers_out", type=str, default="answers.json",
                        help="Path to save output answers.json.")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Minimum word occurrences to be included in vocabulary (default: 2).")
    parser.add_argument("--num_answers", type=int, default=3000,
                        help="Number of most frequent answers to select (default: 3000).")
    parser.add_argument("--fake", action="store_true",
                        help="Build dummy/synthetic vocab and answers for testing before real data exists.")
    args = parser.parse_args()

    print("=" * 60)
    print("VQA Vocabulary and Answer List Builder")
    print("=" * 60)

    if args.fake:
        import random
        print("[FAKE MODE] Generating synthetic question-answer pairs for dry-run testing...")
        rng = random.Random(42)
        sample_words = [
            "what", "color", "is", "the", "cat", "dog", "car", "sky", "shirt",
            "how", "many", "people", "are", "there", "in", "picture", "standing",
            "sitting", "is", "it", "sunny", "rainy", "day", "night", "room", "table"
        ]
        sample_answers = ["yes", "no", "white", "black", "blue", "red", "green", "2", "1", "3", "brown", "yellow"]
        pairs = []
        for _ in range(500):
            q_len = rng.randint(3, 6)
            q_words = [rng.choice(sample_words) for _ in range(q_len)]
            q_text = " ".join(q_words) + " ?"
            ans_text = rng.choice(sample_answers)
            pairs.append((q_text, ans_text))
    else:
        if not os.path.isfile(args.questions):
            print("=" * 60)
            print(f"Error: Questions file not found at: '{args.questions}'")
            print("=" * 60)
            print("To run with your questions file, specify:")
            print("    python data/build_vocab.py --questions path/to/questions.json")
            print("\nTo test with synthetic dummy data before Person A's dataset is ready, run:")
            print("    python data/build_vocab.py --fake")
            print("=" * 60)
            sys.exit(1)

        print(f"Loading questions from: {args.questions}")
        if args.annotations:
            print(f"Loading annotations from: {args.annotations}")
        pairs = load_dataset(args.questions, args.annotations)

    vocab, top_answers, stats = build_vocab_and_answers(
        pairs,
        min_freq=args.min_freq,
        num_answers=args.num_answers
    )

    # Ensure output parent directories exist if specified
    for out_path in [args.vocab_out, args.answers_out]:
        dirname = os.path.dirname(out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # Save vocab.json
    with open(args.vocab_out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary ({stats['vocab_size']} words) to: {args.vocab_out}")

    # Save answers.json (list of strings, index = class ID)
    with open(args.answers_out, "w", encoding="utf-8") as f:
        json.dump(top_answers, f, indent=2)
    print(f"Saved answers list ({stats['num_top_answers']} classes) to: {args.answers_out}")

    # Print required statistics
    print("-" * 60)
    print(f"Dataset size (questions):            {stats['dataset_size']:,}")
    print(f"Number of distinct answers:          {stats['distinct_answers']:,}")
    print(f"Questions answered by top {args.num_answers}:     {stats['covered_questions']:,} / {stats['dataset_size']:,} ({stats['coverage_percentage']:.2f}%)")
    print("-" * 60)


if __name__ == "__main__":
    main()

