# Shared Task Information Science
SemEval 2022 PreTENS (Presupposed Taxonomies: Evaluating Neural Network Semantics)

Excerpt from the task website:
>The PreTENS shared task hosted at SemEval 2022 aims at focusing on semantic competence with specific attention on the evaluation of language models with respect to the  recognition of appropriate taxonomic relations between two nominal arguments (i.e. cases where one is a supercategory of the other, or in extensional terms, one denotes a superset of the other). For instance animal is a supercategory of dog.
 
* Shared task website: [https://sites.google.com/view/semeval2022-pretens/home-page](https://sites.google.com/view/semeval2022-pretens/home-page)
* Dataset repo: [https://github.com/shammur/SemEval2022Task3](https://github.com/shammur/SemEval2022Task3)

## Installation
1. (Optionally) create a virtual environment for this project (tested on python 3.8+)
2. Run `pip install -r requirements.txt`
3. Download an English spacy model for the data generation: `python -m spacy download en_core_web_sm`

## Data generation
For specific data generation scripts and explanations, see the `/data_generation` folder.

## Models
Run `model_experiments/evaluate_models.py` to generate predictions for a given dataset and trained model.

The model we used to generate the submission for the shared task can be downloaded from here: https://drive.google.com/drive/folders/14cSrG_2IyKcTf00C0uRlCqwBFcC87dkO?usp=sharing.
