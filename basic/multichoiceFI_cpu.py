import os
import random
import numpy as np
import torch
import time
import json
from tqdm.auto import tqdm
from lm_eval import evaluator
import lm_eval
import tinyBenchmarks
import argparse

torch.multiprocessing.set_start_method('spawn', force=True)

device = "cpu"
torch.set_printoptions(threshold=np.inf)


def seed_torch(seed=196):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_torch()


def perform_bit_flip_weight(tensor, bit_position):
    """Perform double-bit flip on weight tensor elements."""
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = (1 << bit_position[0]) | (1 << bit_position[1])
        bits = bits ^ mask
        flipped_tensor = bits.view(torch.bfloat16)
    return flipped_tensor


def perform_bit_flip_neuron(tensor, bit_position):
    """Perform double-bit flip on neuron output elements."""
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = (1 << bit_position[0]) | (1 << bit_position[1])
        bits = bits ^ mask
        flipped_tensor = bits.view(torch.bfloat16)
    return flipped_tensor


def perform_bit_flip_single(tensor, bit_position):
    """Perform single-bit flip on neuron output elements."""
    with torch.no_grad():
        tensor_bf16 = tensor.to(torch.bfloat16)
        bits = tensor_bf16.view(torch.int16)
        mask = (1 << bit_position)
        bits = bits ^ mask
        flipped_tensor = bits.view(torch.bfloat16)
    return flipped_tensor


def create_output_hook(module, bit_position, coordinates, token_position, fault_mode):
    """Create a hook function to inject errors into the output tensor."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if not hasattr(hook, 'count'):
            hook.count = 0
        if hook.count == token_position:
            x, y = coordinates
            if x < output.shape[1] and y < output.shape[2]:
                if fault_mode == 'neuron':
                    output[0, x, y] = perform_bit_flip_neuron(output[0, x, y], bit_position)
                elif fault_mode == 'single':
                    output[0, x, y] = perform_bit_flip_single(output[0, x, y], bit_position)
            else:
                print(f"Invalid coordinates: x={x}, y={y}, output shape={output.shape}")
        hook.count += 1
        return output
    return hook


def save_detailed_results(results, filename="weight_bit_flip_detailed_results.json"):
    """Save detailed results to JSON file and properly handle numpy types."""
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif isinstance(obj, tuple):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, torch.dtype):
            return str(obj)
        else:
            return obj

    converted_results = convert_numpy_types(results)
    with open(filename, "w") as f:
        json.dump(converted_results, f, indent=2)

    print(f"\nDetailed results saved to {filename}")


def record_output_dimensions(model, task, device, limit=None):
    """Record output dimensions of the first layer's first module during baseline evaluation."""
    sequence_lengths = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        sequence_lengths.append(output.shape[1])

    first_layer = model.model.model.layers[0]
    module = first_layer.self_attn.v_proj
    hook = module.register_forward_hook(hook_fn)

    evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        batch_size=1,
        num_fewshot=0,
        device=device,
        limit=limit,
    )

    hook.remove()

    min_seq_len = min(sequence_lengths)
    print(f"Observed minimum sequence length: {min_seq_len}")

    return min_seq_len


def main():
    parser = argparse.ArgumentParser(description="Multi-choice fault injection experiment (CPU, Qwen2.5-0.5B)")
    parser.add_argument('--fault_mode', type=str, default='weight', choices=['weight', 'neuron', 'single'],
                        help='Fault injection mode: weight, neuron (double bit), or single (single bit)')
    parser.add_argument('--num_trials', type=int, default=2, help='Number of bit flip trials')
    parser.add_argument('--task', type=str, default='mmlu',
                        choices=['mmlu', 'arc', 'hella', 'wino', 'truth'],
                        help='Evaluation task: mmlu, arc, hella, wino, or truth')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of evaluation samples per run (default: full tiny benchmark)')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")

    output_dir = f"qwen0.5b{args.task}_{args.fault_mode}_cpu"
    os.makedirs(output_dir, exist_ok=True)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # Qwen2.5-0.5B: 24 layers
    layer_weights = {
        'self_attn.v_proj': 1,
        'self_attn.k_proj': 1,
        'self_attn.q_proj': 7,
        'self_attn.o_proj': 7,
        'mlp.up_proj': 37,
        'mlp.gate_proj': 37,
        'mlp.down_proj': 37
    }
    num_layers = 24

    print("Loading model...")
    model = lm_eval.models.huggingface.HFLM(
        pretrained=model_name,
        dtype=torch.bfloat16,
        device=device,
    )
    print(f"Model dtype: {next(model.model.parameters()).dtype}")

    task_mapping = {
        'mmlu': {'name': 'tinyMMLU', 'metric': 'acc_norm,none'},
        'arc': {'name': 'tinyArc', 'metric': 'acc_norm,none'},
        'hella': {'name': 'tinyHellaswag', 'metric': 'acc_norm,none'},
        'wino': {'name': 'tinyWinogrande', 'metric': 'acc_norm,none'},
        'truth': {'name': 'tinyTruthfulQA_mc1', 'metric': 'acc,none'}
    }

    task_info = task_mapping[args.task]
    task = task_info['name']
    metric = task_info['metric']

    if args.fault_mode in ['neuron', 'single']:
        print("Recording output dimensions during baseline evaluation...")
        min_seq_len = record_output_dimensions(model, task, device, limit=args.limit)

    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer] / total_weight for layer in layers]

    results = {
        "base_accuracy": None,
        "base_results_detail": None,
        "bit_flip_trials": [],
    }

    print("Performing baseline evaluation...")
    baseline_results = evaluator.simple_evaluate(
        model=model,
        tasks=[task],
        batch_size=1,
        num_fewshot=0,
        device=device,
        limit=args.limit,
    )

    results["base_accuracy"] = baseline_results["results"][task][metric]
    results["base_results_detail"] = baseline_results
    print(f"Baseline accuracy on {task}: {results['base_accuracy']:.4f}")

    baseline_results_path = os.path.join(output_dir, "baseline_results.json")
    save_detailed_results(baseline_results, baseline_results_path)

    num_trials = args.num_trials
    progress_bar = tqdm(range(num_trials), desc="Bit flip trials")

    for trial in range(num_trials):
        trial_seed = 196 + trial
        seed_torch(trial_seed)
        layer_idx = random.randint(0, num_layers - 1)
        selected_module = random.choices(layers, weights=weights)[0]

        target_layer = model.model.model.layers[layer_idx]
        module_path = selected_module.split('.')
        current_module = target_layer
        for path_part in module_path:
            current_module = getattr(current_module, path_part)

        weight_tensor = current_module.weight

        if args.fault_mode == 'weight':
            x = random.randint(0, weight_tensor.shape[0] - 1)
            y = random.randint(0, weight_tensor.shape[1] - 1)
            bit_position = random.sample(range(16), 2)
            original_value = weight_tensor[x, y].item()
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_weight(weight_tensor[x, y], bit_position)
            flipped_value = weight_tensor[x, y].item()
        else:
            y = random.randint(0, weight_tensor.shape[0] - 1)
            x = random.randint(0, min_seq_len - 1)
            if args.fault_mode == 'neuron':
                bit_position = random.sample(range(16), 2)
            else:
                bit_position = random.randint(0, 15)
            token_position = random.randint(0, min_seq_len - 1)
            hook = create_output_hook(current_module, bit_position, (x, y), token_position, args.fault_mode)
            hook_handle = current_module.register_forward_hook(hook)

        start_time = time.time()
        bit_flip_results = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            batch_size=1,
            num_fewshot=0,
            device=device,
            limit=args.limit,
        )
        end_time = time.time()

        if args.fault_mode == 'weight':
            with torch.no_grad():
                weight_tensor[x, y] = perform_bit_flip_weight(weight_tensor[x, y], bit_position)
        else:
            hook_handle.remove()

        bit_flip_acc = bit_flip_results["results"][task][metric]

        trial_data = {
            "trial_id": trial,
            "layer_idx": layer_idx,
            "module": selected_module,
            "coordinates": {"x": int(x), "y": int(y)},
            "bit_position": bit_position,
            "accuracy": bit_flip_acc,
            "accuracy_change": bit_flip_acc - results["base_accuracy"],
            "time_taken": end_time - start_time,
            "detailed_results": bit_flip_results
        }

        if args.fault_mode == 'weight':
            trial_data.update({
                "original_value": original_value,
                "flipped_value": flipped_value
            })

        results["bit_flip_trials"].append(trial_data)

        trial_results_path = os.path.join(output_dir, f"trial_{trial:03d}_results.json")
        save_detailed_results(trial_data, trial_results_path)

        if (trial + 1) % 10 == 0:
            avg_acc = np.mean([t["accuracy"] for t in results["bit_flip_trials"]])
            print(f"Average accuracy after {trial + 1} trials: {avg_acc:.4f}")
            save_detailed_results(results, os.path.join(output_dir, "all_results_interim.json"))

        progress_bar.update(1)

    avg_accuracy = np.mean([trial["accuracy"] for trial in results["bit_flip_trials"]])
    accuracy_stddev = np.std([trial["accuracy"] for trial in results["bit_flip_trials"]])
    avg_time = np.mean([trial["time_taken"] for trial in results["bit_flip_trials"]])

    print("\n--- Final results ---")
    print(f"Baseline accuracy: {results['base_accuracy']:.4f}")
    print(f"Average bit flip accuracy: {avg_accuracy:.4f} (± {accuracy_stddev:.4f})")
    print(f"Average evaluation time: {avg_time:.2f} seconds")

    module_impacts = {}
    for module in layer_weights.keys():
        module_trials = [t for t in results["bit_flip_trials"] if t["module"] == module]
        if module_trials:
            module_impact = np.mean([t["accuracy"] for t in module_trials])
            module_impacts[module] = module_impact

    print("\n--- Module-type impacts ---")
    for module, impact in module_impacts.items():
        print(f"{module}: {impact:.4f} (difference from baseline: {impact - results['base_accuracy']:.4f})")

    results["summary"] = {
        "avg_accuracy": float(avg_accuracy),
        "accuracy_stddev": float(accuracy_stddev),
        "avg_time": float(avg_time),
        "module_impacts": module_impacts
    }

    save_detailed_results(results, os.path.join(output_dir, "all_results_final.json"))

    summary_results = {
        "base_accuracy": results["base_accuracy"],
        "summary": results["summary"],
        "bit_flip_trials": [{k: v for k, v in trial.items() if k != 'detailed_results'}
                            for trial in results["bit_flip_trials"]]
    }
    save_detailed_results(summary_results, os.path.join(output_dir, "summary_results.json"))

    print("\nAll results saved to output directory")


if __name__ == "__main__":
    main()
