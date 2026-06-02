# Team workflow — Git branching guide

## Branch strategy
```
main          ← stable, working code only. Never commit directly here.
develop       ← integration branch. All features merge here first.
feature/...   ← your daily work branch
```

## Daily workflow for each team member

### Person A (ML engineer)
```bash
git checkout develop
git pull origin develop
git checkout -b feature/train-cardiovascular
# ... do your work ...
git add .
git commit -m "feat: train XGBoost on cardiovascular dataset, AUC 0.88"
git push origin feature/train-cardiovascular
# Open a Pull Request to develop on GitHub
```

### Commit message format
```
feat: add new feature
fix: fix a bug
data: data processing changes
model: model training / evaluation
docs: documentation only
test: add or fix tests
```

## Who owns what
| Person | Files they own |
|--------|---------------|
| A | src/models/, src/data/, notebooks/01-04 |
| B | src/rag/, src/anomaly/, notebooks/05-06 |
| C | app/, tests/, docs/, notebooks/07 |

## Never commit
- .env files (API keys)
- data/raw/* or data/processed/* (too large)
- models/saved/*.pkl (too large — share via Google Drive)
