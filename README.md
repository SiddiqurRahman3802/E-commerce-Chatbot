# E-commerce-Chatbot

## Project Structure

```
E-commerce-Chatbot/
├── configs/                        # Configuration files
│   ├── base_config.yaml            # Base configuration
│   ├── model_config.yaml           # Model-specific config
│   ├── data_prep_config.yaml       # Data preprocessing config
│   ├── evaluation_config.yaml      # Evaluation config
│   └── rag_config.yaml             # RAG setup config (future)
├── data/
│   ├── raw/                        # Original dataset files
│   ├── processed/                  # Cleaned and preprocessed data
│   └── evaluation/                 # Test datasets and evaluation data
├── mlflow/                         # MLflow experiment tracking
│   └── e-commerce-chatbot-finetuning/
├── models/                         # Saved model checkpoints
├── notebooks/                      # Jupyter notebooks for exploration
│   └── chatbot_exploration.ipynb
├── results/                        # Evaluation results
│   ├── fine_tuning_runs/           # Results from fine-tuning
│   └── evaluations/                # BLEU, ROUGE metrics
├── scripts/                        # Runnable scripts
│   ├── data_prep_script.py         # Data preprocessing
│   ├── fine_tuning_script.py       # Fine-tuning the model
│   ├── evaluation_script.py        # Evaluation with metrics
│   └── rag_setup.py                # RAG implementation (future)
├── src/                            # Source code modules
│   ├── data_prep.py                # Data preprocessing
│   ├── fine_tuning.py              # Model fine-tuning
│   ├── evaluation.py               # Evaluation utilities
│   ├── rag_setup.py                # RAG implementation (future)
│   └── utils.py                    # Utility functions
├── utils/                          # Utility modules
│   ├── mlflow_utils.py             # MLflow integration
│   ├── dvc_utils.py                # DVC utilities
│   ├── yaml_utils.py               # Config handling
│   ├── system_utils.py             # System utilities
│   ├── huggingface_utils.py        # HuggingFace integration
│   └── constants.py                # Project constants
├── .dvcignore                      # DVC ignore patterns
├── .gitignore                      # Git ignore patterns
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # DVC pipeline lock file
├── params.yaml                     # DVC parameters
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation               # project overview and instructions
```
