# Shared Task Information Science
PreTENS shared task.

Excerpt from the task website:
>The PreTENS shared task hosted at SemEval 2022 aims at focusing on semantic competence with specific attention on the evaluation of language models with respect to the  recognition of appropriate taxonomic relations between two nominal arguments (i.e. cases where one is a supercategory of the other, or in extensional terms, one denotes a superset of the other). For instance animal is a supercategory of dog.
 
* Shared task website: [https://sites.google.com/view/semeval2022-pretens/home-page](https://sites.google.com/view/semeval2022-pretens/home-page)
* Dataset repo: [https://github.com/shammur/SemEval2022Task3](https://github.com/shammur/SemEval2022Task3)

## Installation
1. (Optionally) create a virtual environment for this project (tested on python 3.8+)
2. Run `pip install -r requirements.txt`
3. Download an English spacy model for the data generation: `python -m spacy download en_core_web_sm`

## Data generation
For specific data generation scripts and explanations, see `data_generation`.

## Models
- Run `model_experiments/run_baselines.py` to run the baselines for a given dataset.
- Run `model_experiments/bert_model_inference.py` to generate predictions for a given dataset.

## TODO
- [x] test set maken (nieuwe templates, nieuwe woordsoorten/onderwerpen) (E)
- [ ] handmatige templates bruikbaar maken (E)
- [x] IT & FR testdata maken (E)
- [x] unieke woorden uit de data halen en in woordenlijsten zetten (G)
- [ ] meer woorden genereren en toevoegen (G)
- [ ] data generatie perfectioneren (G)
- [ ] wordnet 'zij-relaties' eruit halen (eventueel Babelnet -> has kind) (G/F)
- [x] code voor inverten erop zetten voor (tot nu toe voor Engels en Frans) (F)
- [ ] code voor inverten aanvullen met Italiaanse regex patterns (F)
- [ ] eerst focussen op Engels, daarna de nieuwe dataset daarvoor vertalen naar IT en FR (G/W/F/E)
- [x] PEGASUS script erop zetten (W)
- [x] data generen samenvoegen in een script (W)
- [ ] als we de nieuwe datasets hebben -> uitproberen wat het effect is van de datageneratietechnieken (los en gecombineerd)
- [ ] als we verbeteringen zien, Bert model verder trainen op de task ipv embeddings in SVM (peregine uitproberen)
