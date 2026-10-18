"""Run REQ-004 immutable initial analysis / follow-up validation commands."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from master_thesis_modules.evaluation_study.additional_evaluation import main

if __name__ == '__main__':
    main()
