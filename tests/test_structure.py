from pathlib import Path

def test_structure():
    root = Path(__file__).resolve().parents[1]
    required = [
        'README.md',
        'DATA.md',
        'PORTFOLIO.md',
        'ETHICS.md',
        'REPRODUCIBILITY.md',
        'CITATION.cff',
        'src/run_experiment.py',
        'paper/paper.md',
        'assets/01_cover.svg',
        'assets/02_data_pipeline.svg',
        'assets/03_data_or_model.svg',
        'assets/04_evaluation_or_results.svg'
    ]
    for p in required:
        assert (root / p).exists(), p
