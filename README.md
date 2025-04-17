# Formation IA pour la Dynamo de Chambéry

## Installation

Testé avec Python 3.12.9, mais cela doit fonction avec n'importe quelle version de Python > 3.11.
Pour installer Python, voir [www.python.org](www.python.org).

Une fois Python installé (sur Linux ou MacOS, mais c'est similaire sur Windows) :
```shell
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Les différents fichiers peuvent être testés avec `python fichier.py`

Dans le cas case de `06 - finetuning yolo v5 small.py`, il y a une image pour tester que cela fonctionne correctement : `Image_test.jpg`.

## Zoom sur le fine tuning de Yolov5 small

Le code de `06 - finetuning yolo v5 small.py` utilise une version finetunée du modèle Yolov5 Small. Pour ce faire, un dataset de *détritus* récupéré sur [Kaggle](https://www.kaggle.com/datasets/davianmartinovci/litter-detection?resource=download
) a été utilisé.

Le fine tuning de Yolov5 a été réalisé sur une VM avec un GPU assez ancien chez Paperspace (Digital Ocean). Les étapes du finetuning:
- clonage du repo yolov5:
```shell
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```
- copie du dataset dans le répertoire `data` du repo yolov5
- édition du fichier `dataset.yaml` pour ajuster les espaces parasites
- lancement du finetuning :
`python train.py --img 640 --batch 16 --epochs 100 --data data/litter/data/dataset.yaml --weights yolov5s.pt --workers 8`
**Attention à ajuster le nombre de `batch` ou le nombre de `workers` en fonction de la machine**

Pour des modèles plus gros (et même pour Yolov5 Small), il peut être intéressant de geler la partie *backbone* de Yolo en ajoutant `--freeze 10` à la commande ci-dessus. Celà permet d'accélérer les choses. Dans ce cas, réduire le nombre d'`epochs` et reprendre un nouveau finetuning ensuite sans l'arguement `--freeze 10` mais en prenat le modèle qui a été généré précédemment. Pour illustration :
`python train.py --img 640 --batch 16 --epochs 50 --data data/litter/data/dataset.yaml --weights yolov5s.pt --workers 8 --freeze 10`
suivi de :
`python train.py --img 640 --batch 16 --epochs 50 --data data/litter/data/dataset.yaml --weights runs/train/exp/weights/best.pt --workers 8 --hyp runs/train/exp/hyp.yaml`

Il est intéressant de modifier `hyp.yaml`, notamment `lr0` et `lrf` qui sont les *learning rates* de départ et de fin. Dans le contexte de finetuning après avoir déjà réalisé 50 *epochs*, ceux-ci ont vocation à être plus faibles, notamment `lrf`. Par exemple `lr0: 0.01` et `lrf: 0.003`. C'est le concept de *learning rate annealing* qui consiste à réduire le *learning rate* dans le temps pour améliorer les performances et éviter les problèmes de non convergence au niveau de la *back propagation*.
