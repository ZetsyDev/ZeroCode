from datasets import load_dataset
import os
import pandas as pd
import tqdm

class Args:
    eval_n_limit = 100
    eval_num_workers = 10
    eval_output_dir = "gaia_eval_output"
    eval_output_file = "output.jsonl"
    dataset_cache_dir = ".cache"

def prepare_dataset(gaia_tests, args):
    # TODO: Fix this
    gaia_tests = gaia_tests.head(args.eval_n_limit)
    finished_tests = []
    if os.path.exists(args.eval_output_file):
        finished_tests = pd.read_json(args.eval_output_file, lines=True)
    gaia_tests = gaia_tests[~gaia_tests['instance_id'].isin(finished_tests['instance_id'])]
    return gaia_tests

def process_instance(instance: pd.Series, args: Args):
    # TODO: Process the instance here
    instructions = f"Answer the following question based on the context provided.\n\nContext: {instance['context']}\n\nQuestion: {instance['question']}\n\nAnswer:"
    pass

def run_evaluation(dataset: pd.DataFrame, args: Args):
    # TODO: Process the dataset here
    total_instances = len(dataset)
    pbar = tqdm(total=total_instances, desc='Instances processed')
    output_fp = open(args.eval_output_file, 'a')
    for index, row in dataset.iterrows():
        answer = process_instance(row, args)
        instance_id = row['instance_id']
        output_fp.write(f'{{"instance_id": "{instance_id}", "prompt": "{prompt}", "answer": "{answer}"}}\n')
        pbar.update(1)
    output_fp.close()

async def main():
    args = Args()
    dataset_name = "gaia-benchmark/GAIA"
    config_name = "2023_level1"
    dataset = load_dataset(dataset_name, config_name, cache_dir=args.dataset_cache_dir)
    gaia_tests = dataset[args.data_split].to_pandas()
    
    
    output_file = os.path.join(args.eval_output_dir, 'output.jsonl')
    prepared_dataset = prepare_dataset(gaia_tests, output_file, args.eval_n_limit)

    run_evaluation(
        dataset=prepared_dataset,
        args=args
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
