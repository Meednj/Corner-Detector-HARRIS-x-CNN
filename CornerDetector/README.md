# Guide du projet

Ce projet compare deux pipelines de détection de coins :

- `cnn 2.0` : détecteur de coins CNN basé sur PyTorch
- `HarrisDetector` : détecteur de coins Harris classique implémenté en MATLAB
- `app.py` : application Flask qui lance les deux méthodes et sert l’interface dans `index.html`

## Démarrage rapide

Depuis le dossier `CornerDetector` :

```bash
python app.py
```

Ouvrez ensuite l’application dans votre navigateur à l’adresse locale affichée dans le terminal.

L’interface web permet de charger une image, d’exécuter les deux détecteurs et de comparer leurs superpositions ainsi que leurs courbes de scores.

## Utilisation du script CNN

Le script principal d’inférence est :

- `cnn 2.0/predict.py`

Lancez-le ainsi :

```bash
python predict.py --checkpoint runs/corner_detector_best.pt
```

Le script utilise maintenant des valeurs par défaut définies en haut du fichier pour l’image et les paramètres principaux. En haut de `predict.py`, vous pouvez modifier :

- `IMAGE_PATH` pour l’image d’entrée
- `THRESHOLD` pour le seuil CNN
- `NMS_KERNEL` pour la taille du noyau de suppression des maxima locaux

### Comment changer les paramètres CNN

Ouvrez `cnn 2.0/predict.py` et modifiez ces valeurs au début du fichier :

```python
IMAGE_PATH = Path("path/to/godmode.png")
THRESHOLD = -1.0
NMS_KERNEL = 21
```

Signification :

- `IMAGE_PATH` : image utilisée par défaut quand vous lancez le script
- `THRESHOLD = -1.0` : mode automatique
- `THRESHOLD > 0` : seuil manuel
- `NMS_KERNEL` : taille de la fenêtre utilisée pour conserver seulement les pics locaux forts

Vous pouvez aussi les remplacer en ligne de commande si besoin :

```bash
python predict.py --checkpoint runs/corner_detector_best.pt --image path/to/image.png --threshold 0.70 --nms-kernel 15
```

## Rôle de chaque fichier

### Fichiers globaux

- `app.py` : backend Flask. Charge le checkpoint CNN, exécute le détecteur CNN et le détecteur Harris, prépare les images et les résultats, enregistre les sorties combinées et expose la route `/detect` utilisée par l’interface.
- `index.html` : interface web. Permet de charger une image, régler les paramètres, lancer les deux détecteurs et afficher les résultats.

### `cnn 2.0`

- `predict.py` : script autonome d’inférence CNN. Charge une image, exécute le détecteur CNN, applique le seuillage et la NMS, puis enregistre l’image avec les coins détectés.
- `train.py` : script d’entraînement du modèle CNN.
- `demo_synthetic.py` : script de démonstration/test pour les données synthétiques ou les vérifications visuelles.
- `corner_detector/model.py` : définit le modèle CNN et les fonctions de chargement des checkpoints.
- `corner_detector/dataset.py` : utilitaires de dataset et de prétraitement pour l’entraînement et les tests.
- `corner_detector/postprocess.py` : utilitaires de post-traitement comme le seuillage et l’extraction des coins.
- `infer.py.old` : ancienne version de l’inférence conservée comme sauvegarde/référence.
- `requirements.txt` : dépendances Python du projet CNN.
- `runs/` : contient les checkpoints entraînés, par exemple `corner_detector_best.pt`.
- `results/` : enregistre les images générées par le script CNN.
- `path/to/` : dossier de chemin fictif utilisé par les valeurs par défaut du script.

### `HarrisDetector`

- `main.m` : point d’entrée principal MATLAB pour les expériences Harris.
- `harris_detector.m` : implémentation centrale de la détection Harris.
- `sobel_gradients.m` : calcule les gradients d’image.
- `gaussian_kernel.m` : construit les noyaux gaussiens utilisés pour le lissage.
- `non_max_suppression.m` : supprime les réponses faibles et conserve les pics.
- `add_noise.m` : ajoute du bruit aux images de test.
- `change_brightness.m` : ajuste la luminosité pour les expériences.
- `images/` : images d’entrée pour les tests MATLAB.
- `results/` : sorties enregistrées des expériences Harris.
- `notes.txt` : notes et commentaires d’implémentation.
- `script.m.old`, `script_pfe.m.old` : anciennes versions du script conservées en sauvegarde.

## Fonctionnement de l’application complète

1. Vous chargez une image dans l’interface web.
2. `app.py` envoie l’image dans le pipeline CNN.
3. `app.py` lance aussi le détecteur Harris.
4. L’application renvoie :
   - l’image originale
   - la superposition CNN
   - la heatmap CNN
   - la superposition Harris
   - la heatmap Harris
   - le nombre de coins détectés
   - les données de courbe des scores de coins détectés
5. L’interface affiche tout côte à côte pour la comparaison.

## Conseils

- Si le résultat CNN est trop dense, augmentez `THRESHOLD` ou utilisez un seuil manuel.
- Si trop de points proches survivent, augmentez `NMS_KERNEL`.
- Si vous voulez le mode automatique, laissez `THRESHOLD = -1.0`.
- Si `app.py` ne trouve pas le checkpoint, vérifiez que `cnn 2.0/runs/corner_detector_best.pt` existe.

## Notes

- L’application Flask utilise `use_reloader=False` sous Windows pour éviter un double démarrage.
- L’interface et le backend doivent rester synchronisés, donc gardez les champs de réponse dans `app.py` alignés avec ce que `index.html` attend.
