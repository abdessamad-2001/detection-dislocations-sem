# Détection automatique de dislocations émergentes sur images SEM (ECCI)

Ce projet vise à détecter automatiquement les dislocations dans des matériaux III-V sur silicium à l’aide d’un modèle U-Net entraîné sur des images SEM en mode ECCI.

## 📁 Contenu

- `notebooks/` : Notebook principal d'entraînement et d'évaluation du modèle U-Net.
- `src/` : Scripts Python pour la préparation des données, l'annotation interactive et la création de patches.

## 🚀 Usage

Installez les dépendances avec :

```bash
pip install -r requirements.txt
```

Lancez le notebook `unet_dislocations_V2.ipynb` pour entraîner et tester le modèle.

## 🧪 Données

Les images utilisées ne sont pas publiées pour des raisons de confidentialité. Utilisez vos propres images ECCI ou contactez l’auteur pour un accès encadré.

## 🔗 Rapport

Ce dépôt est lié au rapport de stage de master 2 à l’Université Grenoble Alpes (2025).

## 📄 Licence

Ce projet est publié sous la licence MIT (voir fichier LICENSE).
