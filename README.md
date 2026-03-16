## CI/CD Pipeline — MLOps Assignment
This repository contains a GitHub Actions pipeline that automatically validates
the ML environment on every push to any branch except main.
### What the pipeline checks
1. Checkout     — downloads the repo onto the runner VM
2. Set up Python — installs Python 3.10
3. Dependencies — verifies requirements.txt installs cleanly
4. Linter        — runs flake8 to catch code style/syntax errors
5. Dry test      — confirms torch imports and runs successfully
6. Artifact      — saves README.md as 'project-doc' for download
### Trigger logic
The pipeline fires on every push to non-main branches (branches-ignore: main).
This ensures code is tested BEFORE it merges into the official main branch
