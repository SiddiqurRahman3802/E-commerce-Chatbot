<<<<<<< HEAD
# E-commerce-Chatbot
=======
# E-commerce Chatbot Project

## Project Structure

```
├── data/
│   ├── raw/                        # original dataset files (if stored in repo, else only on S3)
│   └── processed/                  # cleaned and preprocessed data (ready for training)
├── notebooks/
│   └── fine_tune_pipeline_colab.ipynb   # Colab notebook with the end-to-end pipeline
├── scripts/                        # Python scripts for each step (if we later convert notebook to scripts)
│   ├── preprocess.py               # script to clean and prepare data
│   ├── train.py                    # script to fine-tune the model (could be run in Colab or locally)
│   ├── evaluate.py                 # script to evaluate the model on a test set
│   ├── upload_to_hub.py            # script using huggingface_hub to push model to HF
│   └── deploy_sagemaker.py         # script to create SageMaker endpoint
├── airflow_dag/                    # (Optional) Airflow DAG definition
│   └── llm_chatbot_pipeline_dag.py # Airflow DAG orchestrating the above scripts
├── mlflow/                         # MLflow configuration or exported runs
│   ├── mlflow.cfg                  # config for tracking server URI, artifact store, etc.
│   └── experiments/                # local MLflow experiment data (if using file store)
├── model_cards/                    # documentation for the model
│   └── README.md                   # Hugging Face model card content
├── frontend/                       # Next.js front-end application
│   ├── pages/
│   │   ├── index.js                # main chat UI page
│   │   └── api/
│   │       └── chat.js             # API route to proxy requests to SageMaker
│   ├── components/                # React components (Chat interface, etc.)
│   ├── public/                    # static assets
│   └── package.json               # front-end project dependencies
└── README.md                       # project overview and instructions
```
>>>>>>> Austin
