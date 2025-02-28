import os
import time
import json
from pathlib import Path
import asyncio
import typer
from tqdm import tqdm
from datasets import load_dataset
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from evaluation.scorer import get_exact_match_score
from zerocode.graph import graph as zerocode_graph

app = typer.Typer()


def save_results(results: list[dict], run_id: str) -> None:
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

    # Merge results, ensuring each task_id appears only once
    task_id_map = {}
    for result in existing_results:
        if "task_id" in result:
            task_id_map[result["task_id"]] = result
    for result in results:
        if "task_id" in result:
            task_id_map[result["task_id"]] = result
    
    # Write combined results
    with open(output_file, "w") as f:
        json.dump(list(task_id_map.values()), f, indent=2)
        
    print(f"Results saved to {output_file}")


def load_results(run_id: int):
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


class GAIAEvaluator:
    """Evaluator class for running evaluations on the GAIA benchmark."""
    
    def __init__(self):
        """Initialize evaluator with agent graph."""
        pass
    
    @staticmethod
    def load_dataset(split="validation", max_samples=100, run_id=int, task_id: str = None):
        """Load the GAIA dataset from Hugging Face."""
        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            name="2023_all", 
            split=split,
            cache_dir=".cache",
            token=os.getenv("HUGGINGFACE_API_KEY")
        )

        dataset = dataset.rename_columns({
            "Question": "question",
            "Final answer": "true_answer",
            "Level": "task"
        })

        # Filter for a specific task if task_id is provided
        if task_id:
            dataset = dataset.filter(lambda x: x["task_id"] == task_id)
        else:
            try:
                results = load_results(run_id)
                ids = set([result["task_id"] for result in results])
                dataset = dataset.filter(lambda x: x["task_id"] not in ids)
            except Exception as e:
                print(f"Error loading results: {e}")
                print("First run, no results to filter")
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
        return dataset
    
    def get_exact_match_score(self, model_answer: str, ground_truth: str) -> bool:
        """Get the score for a question."""
        return get_exact_match_score(model_answer, ground_truth)

    def get_llm_judge_score(self, question: str, model_answer: str, ground_truth: str) -> bool:
        """Get the score for a question."""
        class JudgeResponse(BaseModel):
            correct: bool

        JUDGE_SYSTEM_PROMPT = """Evaluate the given predicted answer to the question by assigning a grade according to the provided grading guidelines.

Grade the predicted answer as "Correct" or "Incorrect" based on these criteria:

- **Correct**: The predicted answer fully contains the ground-truth answer without contradicting the reference answer. Numbers in the predicted answer must match numbers in the true answer exactly (including decimal places). Units can be omitted in the predicted answer.
- **Incorrect**: The predicted answer contradicts the ground-truth answer in any way, even if the contradiction is hedged.

Important: Numbers in the predicted answer must match numbers in the true answer exactly (including decimal places). Units can be omitted in the predicted answer.

# Examples

Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup
True answer: Wout Weghorst

**Correct answers:**
* Wout Weghorst
* Wout Weghorst scored at 83' and 90+11' in that game

**Incorrect answers:**
* Virgil van Dijk
* Virgil van Dijk and Wout Weghorst
* Wout Weghorst and I think van Dijk scored, but I am not totally sure

Question: {question}
True answer: {ground_truth}
Predicted answer: {model_answer}

"""
        llm_judge = ChatAnthropic(model="claude-3-7-sonnet-latest").with_structured_output(JudgeResponse)
        response = llm_judge.invoke(JUDGE_SYSTEM_PROMPT)
        print(f"LLM judge score: {response.correct}")
        return response.correct

    def get_question_score(self, question: str, model_answer: str, ground_truth: str) -> tuple[bool, bool]:
        """Get the score for a question."""
        exact_match = self.get_exact_match_score(model_answer, ground_truth)
        if exact_match:
            return True, True
        else:
            print(f"Exact match failed for question")
            return False, False
            # llm_judge_score = self.get_llm_judge_score(question, model_answer, ground_truth)
            # return False, llm_judge_score


def extract_normalized_answer(answer: str) -> str:
    if "FINAL ANSWER: " in answer:
        return answer[answer.rindex("FINAL ANSWER: ") + len("FINAL ANSWER: "):].strip()
    return answer


async def evaluate_agent_on_gaia(eval_dataset):
    """Run evaluation of the agent on GAIA dataset."""
    results = []
    agent = zerocode_graph
    
    # Run evaluations
    for idx, task in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        task_id = task["task_id"]
        question = task["question"]
        ground_truth = task.get("true_answer", "")
        if ground_truth == "?":
            ground_truth = ""  
        
        try:
            # Run the agent
            metrics = {}
            start_time = time.time()

            file_content = task.get("file_content", None)
            file_path = task.get("file_path", None) 
            file_name = task.get("file_name", None)

            if file_content:
                file_content = task["file_content"]
                question = f"The question is: {question} \n The file content is: {file_content}"
            elif file_path:
                question = f"The question is: {question} \n The file path is: {file_path}"
            elif file_name:
                file_name = task["file_name"]
                question = f"The question is: {question} \n The file name is: {file_name}"

            final_state = await agent.ainvoke({"messages": {"role": "user", "content": question}})
            end_time = time.time()
            time_taken = end_time - start_time

            answer = final_state["messages"][-1].content if isinstance(final_state["messages"][-1], AIMessage) else "No response"
            if isinstance(answer, list):
                answer = answer[1]["text"]
            normalized_answer = extract_normalized_answer(answer)

            metrics[agent.name] = normalized_answer
            metrics["time_taken"] = time_taken
            metrics["solved_by"] = agent.name

            results.append({
                "task_id": task_id,
                "question": question,
                "ground_truth": ground_truth,
                "reasoning_trace": answer,
                "model_answer": normalized_answer,
                "metrics": metrics,
            })
        except Exception as e:
            print(f"Error occurred for task {task_id}: {e}")
    return results


@app.command()
def evaluate(
    max_samples: int = typer.Option(1, help="Maximum number of samples to evaluate"),
    run_id: int = typer.Option(help="Run ID for tracking this evaluation")
):
    """Run evaluation of an agent on the GAIA benchmark."""
    print("Loading GAIA dataset")
    dataset = GAIAEvaluator.load_dataset(max_samples=max_samples, run_id=run_id)
    prepare_workspace(run_id)
    
    print(f"Starting evaluation on {len(dataset)} examples")
    results = asyncio.run(evaluate_agent_on_gaia(dataset))
    save_results(results, run_id)  
    return results


@app.command()
def report(
    run_id: int = typer.Option(help="Run ID to analyze")
):
    """Analyze results from a previous evaluation run."""
    # Load results from the specified run
    exact_correct_count = 0
    llm_judge_correct_count = 0
    incorrect_count = 0
    report = load_results(run_id)
    evaluator = GAIAEvaluator()

    for result in report:
        task_id = result.get("task_id", "")
        if not task_id:
            continue

        if "exact_score" in result and "llm_score" in result:
            exact_score = result["exact_score"]
            llm_score = result["llm_score"]
        else:
            question = result["question"]
            normalized_answer = result.get("model_answer", "")
            ground_truth = result["ground_truth"]
            exact_score, llm_score = evaluator.get_question_score(question, normalized_answer, ground_truth)
            result["exact_score"] = exact_score
            result["llm_score"] = llm_score

        if exact_score:
            exact_correct_count += 1
        elif llm_score:
            llm_judge_correct_count += 1
        else:
            incorrect_count += 1

    save_results(report, run_id)

    # Calculate percentage of correct answers
    total_count = exact_correct_count + llm_judge_correct_count + incorrect_count
    accuracy = (exact_correct_count + llm_judge_correct_count) / total_count * 100
    print(f"\nAccuracy: {accuracy:.2f}% , exact correct: {exact_correct_count}, llm judge correct: {llm_judge_correct_count}, incorrect: {incorrect_count} , total: {total_count}")


@app.command()
def eval_single(
    task_id: str = typer.Option(help="Task ID to retry")
):
    """Try a specific task from a previous evaluation run."""
    # Load the single example from the dataset
    evaluator = GAIAEvaluator()
    prepare_workspace(0)
    dataset = GAIAEvaluator.load_dataset(task_id=task_id)
    
    # Filter for just the requested task
    if len(dataset) == 0:
        print(f"No task found with ID {task_id}")
        return

    print(f"Evaluating {dataset[0]}")
    results = asyncio.run(evaluate_agent_on_gaia(dataset))

    if len(results) == 0:
        print(f"No results found for task {task_id}")
        return

    result = results[0]
    question = result["question"]
    normalized_answer = result["model_answer"]
    ground_truth = result["ground_truth"]
    exact_score, llm_judge_score = evaluator.get_question_score(question, normalized_answer, ground_truth)
    print(f"Exact score: {exact_score}, LLM judge score: {llm_judge_score}")
    print(result)
    return results


def prepare_workspace(run_id: int):
    """Prepare the workspace for a new evaluation run."""
    if run_id:
        output_dir = Path("output") / str(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Make data directory
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Make logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)


@app.command()
def to_jsonl(run_id: int = typer.Option(help="Run ID to analyze")):
    """Convert the evaluation results to a JSONL file."""
    results = load_results(run_id)
    with open(f"output/{run_id}/results.jsonl", "w") as f:
        for result in results:
            d = {"task_id": result["task_id"], "model_answer": result["model_answer"]}
            json.dump(d, f)
            f.write("\n")


# Execute the evaluation if run as a script
if __name__ == "__main__":
    app()