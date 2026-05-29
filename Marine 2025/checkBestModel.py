"""
Inspect a saved ONNX model and its companion JSON results file.
Reports architecture details, parameter counts, and training metrics.

Usage:
    python inspectModel.py                          # Defaults to OUTPUTS/Trained models/bestModel.onnx
    python inspectModel.py path/to/model.onnx       # Custom path
"""

import sys
from pathlib import Path
from json import load as jsonLoad

import onnx
import numpy as np


def inspectONNX(modelPath: Path):
    print(f'\n{"=" * 60}')
    print(f'  ONNX Model Inspection: {modelPath.name}')
    print(f'{"=" * 60}\n')

    model = onnx.load(str(modelPath))
    onnx.checker.check_model(model)
    print('[OK] Model passed ONNX validation.\n')

    # Input / Output shapes.
    print('--- Input / Output ---')
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f'  Input:  {inp.name:>10}  shape = {shape}')
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f'  Output: {out.name:>10}  shape = {shape}')

    # Parameter count.
    print('\n--- Parameters ---')
    totalParams = 0
    layerSummary = {}
    for initializer in model.graph.initializer:
        paramShape = list(initializer.dims)
        paramCount = int(np.prod(paramShape))
        totalParams += paramCount
        # Group by layer prefix (first part of the name before the first dot).
        prefix = initializer.name.split('.')[0] if '.' in initializer.name else initializer.name
        if prefix not in layerSummary:
            layerSummary[prefix] = 0
        layerSummary[prefix] += paramCount

    print(f'  Total parameters: {totalParams:,}')
    print(f'  Total parameters (M): {totalParams / 1e6:.2f}M')
    print(f'\n  Per-layer group:')
    for layerName, count in sorted(layerSummary.items(), key=lambda x: -x[1]):
        print(f'    {layerName:>30}: {count:>12,}  ({count / totalParams * 100:.1f}%)')

    # Opset version.
    print(f'\n--- Opset ---')
    for opset in model.opset_import:
        domain = opset.domain if opset.domain else 'default'
        print(f'  {domain}: version {opset.version}')

    # Node type distribution.
    print(f'\n--- Operator Distribution ---')
    opCounts = {}
    for node in model.graph.node:
        opCounts[node.op_type] = opCounts.get(node.op_type, 0) + 1
    for opType, count in sorted(opCounts.items(), key=lambda x: -x[1]):
        print(f'  {opType:>20}: {count}')

    return totalParams


def inspectResults(resultsPath: Path):
    if not resultsPath.exists():
        print(f'\n[SKIP] No results JSON found at {resultsPath}.')
        return

    print(f'\n{"=" * 60}')
    print(f'  Training Results: {resultsPath.name}')
    print(f'{"=" * 60}\n')

    with open(resultsPath, 'r') as f:
        results = jsonLoad(f)

    print(f'  Trial number:  {results.get("trialNumber", "N/A")}')
    print(f'  Total epochs:  {results.get("maxEpochs", "N/A")}')

    hyperparams = results.get('hyperparameters', {})
    if hyperparams:
        print(f'\n--- Hyperparameters ---')
        for key, value in hyperparams.items():
            if isinstance(value, float):
                print(f'  {key}: {value:.6f}')
            else:
                print(f'  {key}: {value}')

    for split in ['trainingMetrics', 'validationMetrics']:
        metrics = results.get(split, {})
        if metrics:
            label = 'Training' if 'training' in split.lower() else 'Validation'
            print(f'\n--- {label} Metrics (final epoch) ---')
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f'  {key}: {value:.4f}')
                elif isinstance(value, list):
                    formatted = [f'{v:.4f}' for v in value]
                    print(f'  {key}: {formatted}')
                else:
                    print(f'  {key}: {value}')


if __name__ == '__main__':
    # Determine model path.
    if len(sys.argv) > 1:
        modelPath = Path(sys.argv[1])
    else:
        from Various.configurationFile import MODEL_PATH
        modelPath = MODEL_PATH / 'bestModel.onnx'

    if not modelPath.exists():
        print(f'ERROR: Model file not found at {modelPath}')
        sys.exit(1)

    totalParams = inspectONNX(modelPath)

    # Look for companion results JSON.
    resultsPath = modelPath.parent / 'bestResults.json'
    inspectResults(resultsPath)

    print(f'\n{"=" * 60}')
    print(f'  Done.')
    print(f'{"=" * 60}\n')