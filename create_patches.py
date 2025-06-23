#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour créer des patchs à partir d'images ECCI et de leurs masques
pour l'entraînement d'un réseau U-Net.

Ce script découpe les images et masques en patchs de taille fixe,
avec possibilité de chevauchement pour augmenter le nombre d'échantillons.

Auteur: Manus
Date: 2 juin 2025
"""

import os
import numpy as np
import cv2
from skimage import io
import glob
from pathlib import Path
import argparse

def load_image(image_path):
    """
    Charge une image à partir du chemin spécifié.
    
    Args:
        image_path (str): Chemin vers l'image
        
    Returns:
        numpy.ndarray: Image chargée
    """
    try:
        # Charger l'image avec skimage pour gérer différents formats
        image = io.imread(image_path)
        
        # Convertir en niveaux de gris si l'image est en couleur
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        return image
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path}: {e}")
        return None

def create_patches(images_dir, masks_dir, output_dir, patch_size=256, overlap=0.25, mask_suffix="_mask"):
    """
    Crée des patchs à partir des images et masques pour l'entraînement.
    
    Args:
        images_dir (str): Répertoire contenant les images
        masks_dir (str): Répertoire contenant les masques
        output_dir (str): Répertoire de sortie pour les patchs
        patch_size (int): Taille des patchs
        overlap (float): Taux de chevauchement entre les patchs (0-1)
        mask_suffix (str): Suffixe utilisé pour les fichiers de masque
        
    Returns:
        int: Nombre de patchs créés
    """
    # Créer des répertoires pour les patchs
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(os.path.join(patches_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(patches_dir, "masks"), exist_ok=True)
    
    # Récupérer toutes les images
    image_paths = glob.glob(os.path.join(images_dir, "*.tif"))
    
    patch_count = 0
    for image_path in image_paths:
        # Extraire le nom de base de l'image
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Construire le chemin du masque correspondant
        mask_path = os.path.join(masks_dir, f"{base_name}{mask_suffix}.png")
        
        # Vérifier si le masque existe
        if not os.path.exists(mask_path):
            print(f"Masque introuvable pour {base_name}: {mask_path}")
            continue
        
        # Charger l'image et le masque
        image = load_image(image_path)
        mask = load_image(mask_path)
        
        if image is None or mask is None:
            continue
        
        print(f"Traitement de {base_name} - Taille: {image.shape}")
        
        # Calculer le pas pour le chevauchement
        step = int(patch_size * (1 - overlap))
        
        # Extraire des patchs
        for y in range(0, image.shape[0] - patch_size + 1, step):
            for x in range(0, image.shape[1] - patch_size + 1, step):
                # Extraire le patch de l'image
                img_patch = image[y:y+patch_size, x:x+patch_size]
                
                # Extraire le patch du masque
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                
                # Vérifier si le patch du masque contient des dislocations
                # (éviter les patchs vides, sauf pour un certain pourcentage)
                if np.sum(mask_patch) > 0 or np.random.random() < 0.2:
                    # Sauvegarder les patchs
                    patch_img_path = os.path.join(patches_dir, "images", f"patch_{patch_count}.tif")
                    patch_mask_path = os.path.join(patches_dir, "masks", f"patch_mask_{patch_count}.png")
                    
                    cv2.imwrite(patch_img_path, img_patch)
                    cv2.imwrite(patch_mask_path, mask_patch)
                    
                    patch_count += 1
    
    print(f"Créé {patch_count} patchs pour l'entraînement")
    return patch_count

def main():
    """
    Fonction principale pour exécuter le script depuis la ligne de commande.
    """
    parser = argparse.ArgumentParser(description='Création de patchs pour l\'entraînement U-Net')
    parser.add_argument('--images', type=str, required=True, 
                        help='Répertoire contenant les images ECCI')
    parser.add_argument('--masks', type=str, required=True, 
                        help='Répertoire contenant les masques')
    parser.add_argument('--output', type=str, required=True, 
                        help='Répertoire de sortie pour les patchs')
    parser.add_argument('--patch-size', type=int, default=256, 
                        help='Taille des patchs (par défaut: 256)')
    parser.add_argument('--overlap', type=float, default=0.25, 
                        help='Taux de chevauchement entre les patchs (par défaut: 0.25)')
    parser.add_argument('--mask-suffix', type=str, default="_mask", 
                        help='Suffixe utilisé pour les fichiers de masque (par défaut: _mask)')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(args.output, exist_ok=True)
    
    # Créer les patchs
    create_patches(args.images, args.masks, args.output, 
                  args.patch_size, args.overlap, args.mask_suffix)

if __name__ == "__main__":
    # ====== À MODIFIER ICI : CHEMINS ET PARAMÈTRES ======
    images_dir = "resultat/images_2"
    masks_dir = "resultat/masks_2"
    output_dir = "resultat/patches_2"
    patch_size = 256
    overlap = 0.25
    mask_suffix = "_mask"

    # Appel direct de la fonction (sans argparse)
    create_patches(images_dir, masks_dir, output_dir, patch_size, overlap, mask_suffix)