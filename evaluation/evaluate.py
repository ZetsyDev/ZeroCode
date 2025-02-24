import os
import re
import string
from typing import Dict, List, Any
import json
from langchain_anthropic import ChatAnthropic
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pathlib import Path
import typer
from loguru import logger
    
app = typer.Typer()


def save_results(results: Dict, run_id: str) -> None:
    """Save evaluation results to a JSON file in the output directory.
    
    Args:
        results: Dictionary containing evaluation results
        run_id: ID of the evaluation run
    """
    output_dir = Path("output") / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "gaia_evaluation_results.json"
    
    # Load existing results if file exists
    existing_results = []
    if output_file.exists():
        with open(output_file) as f:
            existing_results = json.load(f)
            
    # Append new results
    if isinstance(existing_results, list):
        existing_results.extend(results)
    else:
        existing_results = results
        
    # Write combined results
    with open(output_file, "w") as f:
        json.dump(existing_results, f, indent=2)
        
    print(f"Results saved to {output_file}")

def load_results(run_id: int) -> Dict:
    """Load evaluation results from a JSON file.
    
    Args:
        run_id: ID of the evaluation run to load
        
    Returns:
        Dictionary containing the evaluation results
        
    Raises:
        FileNotFoundError: If no results file exists for the given run_id
    """
    output_dir = Path("output") / str(run_id)
    results_file = output_dir / "gaia_evaluation_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"No results found for run {run_id}")
        
    with open(results_file) as f:
        results = json.load(f)
        
    return results

def remove_boxed(text):
    # Replace \boxed{number} with just the number
    return re.sub(r"\\boxed\{(\d+)\}", r"\1", text)

def normalize_number_str(number_str: str) -> float:
    number_str = remove_boxed(number_str)
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")

def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)

def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


class GAIAEvaluator:
    """Evaluator class for running evaluations on the GAIA benchmark."""
    
    def __init__(self):
        """Initialize evaluator with agent graph.
        """
        pass
    
    @staticmethod
    def load_dataset(split="validation", max_samples=100, run_id=int):
        """Load the GAIA dataset from Hugging Face."""
        dataset = load_dataset("gaia-benchmark/GAIA",name="2023_level1", split=split, cache_dir=".cache", token=os.getenv("HUGGINGFACE_API_KEY"))

        dataset = dataset.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task", "task_id": "id"})
        try:
            results = load_results(run_id)
            ids = set([result["id"] for result in results])
            dataset = dataset.filter(lambda x: x["id"] not in ids)
        except:
            print("First run, no results to filter")
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        return dataset
    
    def create_agent() -> StateGraph:
        """Create a LangGraph agent with the given LLM and tools."""
        llm = ChatAnthropic(model="claude-3-7-sonnet-latest")
        GAIA_NORMALIZATION_PROMPT = """
    Finish your answer with the following template:
    FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
    If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        """
        
        tools = [
        ]
        agent = create_react_agent(llm, tools, prompt=GAIA_NORMALIZATION_PROMPT)
        return agent
        
    def get_question_score(self, model_answer: str, ground_truth: str) -> bool:
        
        if is_float(ground_truth):
            normalized_answer = normalize_number_str(str(model_answer))
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):  # if gt is a list
            # question with the fish: normalization removes punct
            gt_elems = split_string(ground_truth)
            ma_elems = split_string(model_answer)

            if len(gt_elems) != len(ma_elems):  # check length is the same
                logger.debug("Answer lists have different lengths, returning False.", UserWarning)
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):  # compare each element as float or str
                if is_float(gt_elem):
                    normalized_ma_elem = normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    # we do not remove punct since comparisons can include punct
                    comparisons.append(
                        normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                    )
            return all(comparisons)

        else:  # if gt is a str
            return normalize_str(model_answer) == normalize_str(ground_truth)



def extract_normalized_answer(answer: str) -> str:
    if "FINAL ANSWER: " in answer:
        return answer[answer.rindex("FINAL ANSWER: ") + len("FINAL ANSWER: ") :].strip()
    return answer



# 6. Main evaluation function
def evaluate_agent_on_gaia(eval_dataset):
    """Run evaluation of the agent on GAIA dataset."""
    results = []
    evaluator = GAIAEvaluator()
    agent_graph = GAIAEvaluator.create_agent()
    # Run evaluations
    for idx, example in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        id = example["id"]
        question = example["question"]
        ground_truth = example.get("true_answer", "")
        if ground_truth == "?":
            ground_truth = ""  
        
        try:
            # Run the agent
            final_state = agent_graph.invoke({"messages": {"role": "user", "content": question}})
            answer = final_state["messages"][-1].content if isinstance(final_state["messages"][-1], AIMessage) else "No response"
            normalized_answer = extract_normalized_answer(answer)
            correct = evaluator.get_question_score(normalized_answer, ground_truth)
            results.append({
                "id": id,
                "question": question,
                "ground_truth": ground_truth,
                "agent_answer": answer,
                "final_answer": normalized_answer,
                "correct": correct
            })
            # if not correct:
            #     logger.error(f"Incorrect answer for question {id}. \nQuestion: {question}\nGround truth: {ground_truth}\nAgent answer: {answer}\nFinal answer: {normalized_answer}")
            #     break
        
        except Exception as e:
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "agent_answer": "Error occurred",
                "metrics": {"error": str(e)}
            })
    
    return results

def evaluate_agent(max_samples=50, run_id=int):
    """Run the evaluation step."""
    
    
    print("Loading GAIA dataset")
    dataset = GAIAEvaluator.load_dataset(max_samples=max_samples, run_id=run_id)
    
    print(f"Starting evaluation on {len(dataset)} examples")
    results = evaluate_agent_on_gaia(dataset)
    return results, run_id



@app.command()
def evaluate(
    model_name: str = typer.Option("anthropic/claude-3-5-sonnet-20241022", help="Name of the model to evaluate"),
    max_samples: int = typer.Option(1, help="Maximum number of samples to evaluate"),
    run_id: int = typer.Option(help="Run ID for tracking this evaluation")
):
    """Run evaluation of an agent on the GAIA benchmark."""
    results, run_id = evaluate_agent(
        max_samples=max_samples, 
        run_id=run_id
    )
    save_results(results, run_id)  
    return results


@app.command()
def report(
    run_id: int = typer.Option(help="Run ID to analyze")
):
    """Analyze results from a previous evaluation run."""
    # Load results from the specified run
    correct_count = 0
    incorrect_count = 0
    report = load_results(run_id)
    for result in report:
        correct = result["correct"]
        if correct:
            correct_count += 1
        else:
            incorrect_count += 1

    # Calculate percentage of correct answers
    total_count = correct_count + incorrect_count
    accuracy = (correct_count / total_count) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")




# Execute the evaluation if run as a script
if __name__ == "__main__":
    app()