from pathlib import Path
def test_structure():
 root=Path(__file__).resolve().parents[1]
 required=['README.md','DATA.md','PORTFOLIO.md','ETHICS.md','REPRODUCIBILITY.md','CITATION.cff','src/run_experiment.py','paper/paper.md','assets/01_cover.png','assets/02_data_pipeline.png','assets/03_data_or_model.png','assets/04_evaluation_or_results.png']
 for p in required: assert (root/p).exists(),p
