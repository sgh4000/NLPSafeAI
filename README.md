# NLPSafeAI
This is the GitHub repository for the CDT-D2AIR group working on the depression text classification NLP problem

# Requirements to run

**Please note** that there are two separate requirements.txt files, required to run different parts of the code.
- **src/requirements.txt** for running the base training, adversarial training and semantic robustness analysis and verification
- **src/xai_verification/requirements_xai_verification.txt** for running the XAI-verification pipeline.

This was necessary due to conflicting tensor flow versions required for training and SHAP analysis.

# Dataset citation

Depression: Reddit Dataset (Cleaned)
~7000 Cleaned Reddit Labelled Dataset on Depression
https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned/data
License
CC0: Public Domain

# Related work

This project builds on the NLP verification work done by Casadio et al. 

[1] M. Casadio et al., “NLP verification: towards a general methodology for certifying robustness,” European Journal of Applied Mathematics, vol. 37, no. 1, pp. 180–237, 2026. doi:10.1017/S0956792525000099