import re
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NORMALIZED_DENOM = 100000
OUTPUT_TXT = os.path.join(SCRIPT_DIR, 'output_type.txt')
OUTPUT_PDF = os.path.join(SCRIPT_DIR, 'figs', 'output_type.pdf')

MODEL_LABELS = ['Falcon3-7B', 'Qwen2.5-7B']
MODEL_KEYS = ['falcon', 'qwen']

# Folder keys -> chart labels (bar order left to right: 1bit-comp, 2bits-comp, 2bits-mem)
FAULT_META = {
    'single': {'label': '1bit-comp', 'pattern': ''},
    'neuron': {'label': '2bits-comp', 'pattern': '||||'},
    'weight': {'label': '2bits-mem', 'pattern': '/////'},
}
PLOT_FAULT_ORDER = ['single', 'neuron', 'weight']

ANALYSIS_GROUPS = [
    {'model': 'falcon', 'fault_model': 'weight', 'folders': [
        'mfalcongsm8k', 'mfalcongsm8k1', 'mfalcongsm8k2', 'mfalcongsm8k3', 'mfalcongsm8k4',
    ]},
    {'model': 'falcon', 'fault_model': 'neuron', 'folders': [
        'mfalcongsm8kneuron', 'mfalcongsm8kneuron1', 'mfalcongsm8kneuron2',
        'mfalcongsm8kneuron3', 'mfalcongsm8kneuron4',
    ]},
    {'model': 'falcon', 'fault_model': 'single', 'folders': [
        'mfalcongsm8ksingle', 'mfalcongsm8ksingle1', 'mfalcongsm8ksingle2',
        'mfalcongsm8ksingle3', 'mfalcongsm8ksingle4',
    ]},
    {'model': 'qwen', 'fault_model': 'weight', 'folders': [
        'mqwengsm8k', 'mqwengsm8k1', 'mqwengsm8k2', 'mqwengsm8k3', 'mqwengsm8k4',
    ]},
    {'model': 'qwen', 'fault_model': 'neuron', 'folders': [
        'mqwengsm8kneuron', 'mqwengsm8kneuron1', 'mqwengsm8kneuron2',
        'mqwengsm8kneuron3', 'mqwengsm8kneuron4',
    ]},
    {'model': 'qwen', 'fault_model': 'single', 'folders': [
        'mqwengsm8ksingle', 'mqwengsm8ksingle1', 'mqwengsm8ksingle2',
        'mqwengsm8ksingle3', 'mqwengsm8ksingle4',
    ]},
]


def extract_last_number(text):
    if text is None:
        return None
    numbers = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    if numbers:
        last_number = None
        for num in reversed(numbers):
            if num[0] or num[1]:
                last_number = num[0] if num[0] else num[1]
                break
        if last_number:
            cleaned = last_number.replace('$', '').replace(',', '')
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None


def has_infinite_loop(text):
    if text is None:
        return True

    special_char_count = len(re.findall(r'[^\w\s]', text))
    total_length = len(text) if text else 0
    if total_length > 0 and special_char_count / total_length > 0.6:
        return True

    pattern_len = min(10, len(text) // 4) if text else 0
    if pattern_len > 2:
        for i in range(2, pattern_len + 1):
            pattern = text[:i]
            repeats = text.count(pattern)
            if repeats > 10 and repeats * len(pattern) > len(text) * 0.1:
                return True
    return False


def analyze_different_answers(file_path):
    result_changed = 0
    infinite_loop = 0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    for sample in content.split('=' * 80):
        if not sample.strip():
            continue

        baseline_match = re.search(r'Baseline: ([\s\S]*?)(?=\nBit-flip:|$)', sample)
        bitflip_match = re.search(r'Bit-flip: ([\s\S]*?)(?=\nCorrect:|$)', sample)
        if not (baseline_match and bitflip_match):
            continue

        baseline_number = extract_last_number(baseline_match.group(1))
        bitflip_text = bitflip_match.group(1)
        bitflip_number = extract_last_number(bitflip_text)

        if has_infinite_loop(bitflip_text):
            infinite_loop += 1
        elif baseline_number != bitflip_number:
            result_changed += 1

    return result_changed, infinite_loop


def analyze_group(folders, base_dir=SCRIPT_DIR):
    result_changed = 0
    infinite_loop = 0
    for folder in folders:
        file_path = os.path.join(base_dir, folder, 'different_answers.txt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
        rc, il = analyze_different_answers(file_path)
        result_changed += rc
        infinite_loop += il
    return result_changed, infinite_loop


def normalized_rate(count):
    return count / NORMALIZED_DENOM * 100


def collect_stats():
    # stats[model_key][fault_key] = (result_changed_pct, infinite_loop_pct)
    stats = {model: {} for model in MODEL_KEYS}
    rows = []

    for group in ANALYSIS_GROUPS:
        model = group['model']
        fault = group['fault_model']
        print(f'Analyzing: {model} / {fault} ...', flush=True)

        rc, il = analyze_group(group['folders'])
        rc_pct = normalized_rate(rc)
        il_pct = normalized_rate(il)
        stats[model][fault] = (rc_pct, il_pct)
        rows.append([model, fault, f'{rc_pct:.2f}%', f'{il_pct:.2f}%'])

    return stats, rows


def save_summary(rows):
    headers = ['Model', 'Fault Model', 'Result Changed (/100k)', 'Infinite Loop (/100k)']
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells):
        return ' | '.join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    lines = [
        '=' * 70,
        'Bit-flip Summary (denominator = 100,000)',
        '=' * 70,
        fmt_row(headers),
        fmt_row(['-' * w for w in col_widths]),
    ]
    lines.extend(fmt_row(row) for row in rows)

    text = '\n'.join(lines) + '\n'
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f'Summary saved to: {OUTPUT_TXT}')


def plot_stats(stats):
    mpl.rcParams['hatch.linewidth'] = 0.5
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=(2.6, 1))

    bar_width = 0.12
    group_spacing = 0.6
    model_centers = np.arange(len(MODEL_LABELS)) * group_spacing
    bar_offsets = np.array([-bar_width, 0, bar_width])

    base_color_light = 'darksalmon'
    base_color_dark = 'teal'

    all_totals = []
    for model_idx in range(len(MODEL_LABELS)):
        model_center = model_centers[model_idx]

        for fault_idx, fault_key in enumerate(PLOT_FAULT_ORDER):
            meta = FAULT_META[fault_key]
            subtly_value, distorted_value = stats[MODEL_KEYS[model_idx]][fault_key]
            pos = model_center + bar_offsets[fault_idx]
            all_totals.append(subtly_value + distorted_value)

            bar_bottom = ax.bar(
                pos, subtly_value, width=bar_width,
                color=base_color_light, edgecolor='black',
                linewidth=1, hatch=meta['pattern'], zorder=5,
            )
            bar_top = ax.bar(
                pos, distorted_value, width=bar_width,
                bottom=subtly_value, color=base_color_dark,
                edgecolor='black', linewidth=1, hatch=meta['pattern'], zorder=5,
            )

            if model_idx == 0 and fault_idx == 0:
                bar_bottom[0].set_label('Subtly')
                bar_top[0].set_label('Distorted')

    ax.set_xlabel('Models', labelpad=3)
    ax.set_xticks(model_centers, MODEL_LABELS, ha='center')
    ax.set_ylabel('Percentage\nof Outputs (%)')

    max_total = max(all_totals)
    ax.set_ylim(0, max_total * 1.15)
    if max_total <= 20:
        ax.set_yticks([int(max_total * 0.4), int(max_total * 0.8)])
    else:
        ax.set_yticks([int(max_total * 0.3), int(max_total * 0.7)])

    # Same order and hatch binding as bars (left -> right).
    legend_patterns = [
        Patch(
            facecolor=base_color_light, edgecolor='black',
            hatch=FAULT_META[fault_key]['pattern'],
            label=FAULT_META[fault_key]['label'],
        )
        for fault_key in PLOT_FAULT_ORDER
    ]
    legend_colors = [
        Patch(facecolor=base_color_light, edgecolor='black', label='Subtly'),
        Patch(facecolor=base_color_dark, edgecolor='black', label='Distorted'),
    ]

    legend1 = ax.legend(
        handles=legend_patterns, loc='lower left',
        bbox_to_anchor=(-0.08, 1, 1.12, 0.15), mode='expand',
        fontsize='small', ncol=3,
    )
    legend2 = ax.legend(
        handles=legend_colors, loc='lower left',
        bbox_to_anchor=(0, 0.73, 0.66, 0.15), mode='expand',
        fontsize='small', ncol=2,
    )
    ax.add_artist(legend1)

    ax.margins(x=0.1)
    ax.grid(True, axis='y')

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    fig.savefig(
        OUTPUT_PDF, dpi=100, bbox_inches='tight',
        bbox_extra_artists=[legend1, legend2], pad_inches=0.08,
    )
    plt.close(fig)
    print(f'Figure saved to: {OUTPUT_PDF}')


def main():
    stats, rows = collect_stats()
    save_summary(rows)
    plot_stats(stats)


if __name__ == '__main__':
    main()
