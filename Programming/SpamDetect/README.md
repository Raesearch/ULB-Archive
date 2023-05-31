# Detecteur de Spam (Spam detector)

Vous trouverez ci-dessous les instructions et détails sur l'application de détecteur de Spam.
Le but de cette application étant de determiner si un texte donné, venant typiquement d'un email
est catégorisé comme spam ou non.

L'application présenté ici utilise un algorithme d'apprentissage du type 
classification naïve bayésienne (Naive Bayes classifier).

## Installation

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer l'application dans un des trois modes:
le mode `show` (`-s`) vous permettra de visualiser un nuage de mots des spams
contenus dans le fichier spam.csv;

```bash
poetry run python main.py -s
```

le mode `test` (`-t`) vous donnera les métriques de résultats d'un test de
classification de messages aléatoires après entrainement;

```bash
poetry run python main.py -t -d dataset -m model1 model2 ...
```
dataset : this is for choosing the data set used for training
- spam is a dataset made of mails that are either spam or ham
- clickbait are made of headlines, and the classification is about either the title is clickbait or not
- revieuw is made of movies critics made on rotten tomatos, and the classification is about the feeling of the critics, postive or genative

model : le choix du model
-random :  assigne la class au hazard
-bow : naive bayesian with document vectors defined by the bag of words score
-naive : naive bayesian with document vectors defined by the tf-idf score
-knn : K-Nearest Neighbors, documents-word vectors are represented with tf-idf score
-knn-cwc : K-Nearest Neighbors, a confidence weight score is used instead of a tf-idf
-svm : Support Vector Machine
-swm-cwc : Support Vector Machine, with a confidence weight score instead of a tf-idf
-ga : genetic algorithm

le mode `classify` (`-c`) vous permettre de tester une phrase pour savoir quelle sera la classe prédite. Le résultat dépend largement du data set choisie pour l'entrainement.


```bash
poetry run python main.py -c "Can machines think?"
```

Vous verez alors apparaitre dans le terminal la mention
`Spam? : True`, ou `Spam? : False`, suivant si votre message est classé comme
indésirable ou non.

En résumé:
```
usage: main.py [-h] [-s] [-t] [-c CLASSIFY]

Spam detector.

optional arguments:
  -h, --help            show this help message and exit
  -s, --show            Shows the occurence of words as a wordcloud
  -t, --test            trains and tests the algorithms and gives results in différent metrics
  -c CLASSIFY, --classify CLASSIFY
                        Classifies the given text into spam or not spam using TFxIDF
```

![spam screen](../assets/img/spam.png)

[ia-gh]: https://github.com/iridia-ulb/AI-book
