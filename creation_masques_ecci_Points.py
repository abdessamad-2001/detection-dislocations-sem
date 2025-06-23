# -*- coding: utf-8 -*-
"""
Script pour la détection automatique de dislocations sous forme de points
dans les images ECCI de matériaux III-V sur silicium.

Modifié pour détecter uniquement les dislocations comme des points,
en supprimant la détection de lignes.

Auteur: ST Foundation (modifié par Manus)
Date: 3 juin 2025
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, filters, feature, morphology, exposure, util
from skimage.feature import blob_log, peak_local_max
import glob
from pathlib import Path

# For Jupyter notebook compatibility
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    print("Warning: ipywidgets not available. Interactive widgets will not work.")

# Configuration pour afficher les images en plus grand
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

if __name__ == "__main__":
    # Définir les répertoires d'entrée et de sortie
    input_dir = "H:/STAGE LTMlab/Images ECCI/images"
    output_dir = "H:/STAGE LTMlab/Images ECCI/masks"

    # Créer les répertoires de sortie s'ils n'existent pas
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    main()

def load_image(image_path):
    """
    Charge une image ECCI à partir du chemin spécifié.
    
    Args:
        image_path (str): Chemin vers l'image ECCI
        
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

def preprocess_image(image, gaussian_sigma=1.0):
    """
    Prétraite l'image ECCI pour améliorer la détection des dislocations.
    
    Args:
        image (numpy.ndarray): Image ECCI originale
        gaussian_sigma (float): Sigma pour le filtre gaussien
        
    Returns:
        numpy.ndarray: Image prétraitée
    """
    # Normalisation d'histogramme pour améliorer le contraste
    image_eq = exposure.equalize_hist(image)
    
    # Conversion en uint8 pour compatibilité avec OpenCV
    image_eq = util.img_as_ubyte(image_eq)
    
    # Filtrage gaussien pour réduire le bruit
    image_filtered = cv2.GaussianBlur(image_eq, (0, 0), gaussian_sigma)
    
    return image_filtered

def detect_dislocation_points(image, params):
    """
    Détecte les dislocations comme points dans l'image prétraitée.
    
    Args:
        image (numpy.ndarray): Image ECCI prétraitée
        params (dict): Paramètres de détection
        
    Returns:
        tuple: (masque binaire des points détectés, liste des points détectés)
    """
    # Initialiser le masque
    mask = np.zeros_like(image)
    
    # Seuillage adaptatif pour détecter les structures sombres
    thresh = cv2.adaptiveThreshold(image, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 
                                  params['adaptive_block_size'], 
                                  params['adaptive_c'])
    
    # Opérations morphologiques pour nettoyer l'image binaire
    kernel = np.ones((params['morph_kernel_size'], params['morph_kernel_size']), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Trouver les contours des points
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours par taille et circularité
    points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if params['min_area'] <= area <= params['max_area']:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity >= params['min_circularity']:
                    # Calculer le centre du contour
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        radius = int(np.sqrt(area/np.pi))
                        points.append((cx, cy, radius))
                        # Dessiner le point sur le masque
                        cv2.circle(mask, (cx, cy), radius, 255, -1)
    
    return mask, points

def detect_dislocation_points_log(image, params):
    """
    Détecte les dislocations comme points en utilisant Laplacian of Gaussian (LoG).
    
    Args:
        image (numpy.ndarray): Image ECCI prétraitée
        params (dict): Paramètres de détection
        
    Returns:
        tuple: (masque binaire des points détectés, liste des points détectés)
    """
    # Initialiser le masque
    mask = np.zeros_like(image)
    
    # Normaliser l'image pour LoG (0-1)
    image_norm = image.astype(float) / 255.0
    
    # Inverser l'image car LoG détecte les points clairs
    image_inv = 1.0 - image_norm
    
    # Détecter les blobs avec LoG
    blobs = blob_log(image_inv, 
                    min_sigma=params['min_sigma'], 
                    max_sigma=params['max_sigma'], 
                    num_sigma=params['num_sigma'], 
                    threshold=params['threshold'], 
                    overlap=params['overlap'])
    
    # Convertir les résultats en points (x, y, rayon)
    points = []
    for blob in blobs:
        y, x, sigma = blob
        radius = int(sigma * np.sqrt(2))
        points.append((int(x), int(y), radius))
        # Dessiner le point sur le masque
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    
    return mask, points

def detect_dislocation_points_peak(image, params):
    """
    Détecte les dislocations comme points en utilisant les maxima locaux.
    
    Args:
        image (numpy.ndarray): Image ECCI prétraitée
        params (dict): Paramètres de détection
        
    Returns:
        tuple: (masque binaire des points détectés, liste des points détectés)
    """
    # Initialiser le masque
    mask = np.zeros_like(image)
    
    # Inverser l'image car nous cherchons des minima (points sombres)
    image_inv = 255 - image
    
    # Trouver les maxima locaux
    coordinates = peak_local_max(image_inv, 
                                min_distance=params['min_distance'],
                                threshold_abs=params['threshold_abs'] * 255,
                                threshold_rel=params['threshold_rel'],
                                exclude_border=params['exclude_border'])
    
    # Convertir les coordonnées en points (x, y, rayon)
    points = []
    for y, x in coordinates:
        # Utiliser un rayon fixe pour les maxima locaux
        radius = params['point_radius']
        points.append((x, y, radius))
        # Dessiner le point sur le masque
        cv2.circle(mask, (x, y), radius, 255, -1)
    
    return mask, points

def interactive_annotation_simple(image, initial_mask=None):
    """
    Version simplifiée de l'annotation manuelle qui utilise uniquement matplotlib.
    Cette version est plus compatible avec Jupyter Notebook et Spyder.
    
    Args:
        image (numpy.ndarray): Image ECCI
        initial_mask (numpy.ndarray, optional): Masque initial
        
    Returns:
        numpy.ndarray: Masque annoté manuellement
    """
    if initial_mask is None:
        # Créer un masque vide
        mask = np.zeros_like(image)
    else:
        # Utiliser le masque fourni
        mask = initial_mask.copy()
    
    print("\n" + "="*80)
    print("INTERFACE D'ANNOTATION MANUELLE")
    print("="*80)
    print("INSTRUCTIONS:")
    print("• Cliquez et faites glisser sur l'image de DROITE pour dessiner")
    print("• Touches du clavier:")
    print("  - 'e' : Basculer entre DESSINER et EFFACER")
    print("  - '+' : Augmenter la taille du pinceau")
    print("  - '-' : Diminuer la taille du pinceau") 
    print("  - 'c' : Effacer tout le masque")
    print("  - 's' : Sauvegarder et terminer")
    print("• Fermez la fenêtre pour terminer")
    print("="*80)
    
    # Créer une figure pour afficher l'image et le masque
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Afficher l'image originale
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Image originale\n(référence)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Créer une superposition pour mieux voir les modifications
    overlay = np.zeros((image.shape[0], image.shape[1], 3))
    overlay[:, :, 0] = image / 255.0  # Canal rouge
    overlay[:, :, 1] = image / 255.0  # Canal vert  
    overlay[:, :, 2] = image / 255.0  # Canal bleu
    
    # Ajouter le masque en rouge semi-transparent
    mask_colored = np.zeros_like(overlay)
    mask_colored[mask > 0, 0] = 1.0  # Rouge pour les dislocations
    overlay = 0.7 * overlay + 0.3 * mask_colored
    
    overlay_display = axes[1].imshow(overlay)
    axes[1].set_title('Superposition\n(Image + Masque)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Afficher le masque seul
    mask_display = axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Masque\n(Zone de dessin)', fontsize=12, fontweight='bold', color='red')
    axes[2].axis('off')
    
    # Variables pour le dessin
    drawing = False
    last_x, last_y = -1, -1
    brush_size = 8
    erase_mode = False
    finished = False
    
    # Fonction pour mettre à jour l'affichage
    def update_display():
        nonlocal overlay, mask_colored
        # Mettre à jour la superposition
        overlay = np.zeros((image.shape[0], image.shape[1], 3))
        overlay[:, :, 0] = image / 255.0
        overlay[:, :, 1] = image / 255.0
        overlay[:, :, 2] = image / 255.0
        
        mask_colored = np.zeros_like(overlay)
        mask_colored[mask > 0, 0] = 1.0
        overlay = 0.7 * overlay + 0.3 * mask_colored
        
        overlay_display.set_data(overlay)
        mask_display.set_data(mask)
        fig.canvas.draw_idle()
    
    # Fonction pour le dessin sur le masque
    def draw(event):
        nonlocal drawing, last_x, last_y, mask
        
        if not drawing or event.inaxes != axes[2]:
            return
        
        if event.xdata is None or event.ydata is None:
            return
            
        # Coordonnées actuelles
        x, y = int(event.xdata), int(event.ydata)
        
        # Vérifier que les coordonnées sont dans les limites de l'image
        if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
            return
        
        # Si c'est le premier point, initialiser last_x et last_y
        if last_x == -1:
            last_x, last_y = x, y
        
        # Dessiner une ligne entre le dernier point et le point actuel
        cv2.line(mask, 
                (last_x, last_y), 
                (x, y), 
                0 if erase_mode else 255, 
                brush_size)
        
        # Mettre à jour l'affichage
        update_display()
        
        # Mettre à jour les dernières coordonnées
        last_x, last_y = x, y
    
    # Gestionnaires d'événements
    def on_press(event):
        nonlocal drawing, last_x, last_y
        if event.inaxes == axes[2]:
            drawing = True
            last_x, last_y = -1, -1
            draw(event)
    
    def on_release(event):
        nonlocal drawing
        drawing = False
        last_x, last_y = -1, -1
    
    def on_motion(event):
        draw(event)
    
    def on_key(event):
        nonlocal erase_mode, brush_size, finished
        if event.key == 'e':
            erase_mode = not erase_mode
            mode_text = 'EFFACER' if erase_mode else 'DESSINER'
            print(f"🔄 Mode changé: {mode_text}")
            axes[2].set_title(f'Masque - Mode: {mode_text}\n(Taille pinceau: {brush_size})', 
                            fontsize=12, fontweight='bold', 
                            color='blue' if erase_mode else 'red')
            fig.canvas.draw_idle()
        elif event.key == '+' or event.key == '=':
            brush_size = min(brush_size + 2, 30)
            print(f"🖌️ Taille pinceau: {brush_size}")
            mode_text = 'EFFACER' if erase_mode else 'DESSINER'
            axes[2].set_title(f'Masque - Mode: {mode_text}\n(Taille pinceau: {brush_size})', 
                            fontsize=12, fontweight='bold',
                            color='blue' if erase_mode else 'red')
            fig.canvas.draw_idle()
        elif event.key == '-':
            brush_size = max(brush_size - 2, 1)
            print(f"🖌️ Taille pinceau: {brush_size}")
            mode_text = 'EFFACER' if erase_mode else 'DESSINER'
            axes[2].set_title(f'Masque - Mode: {mode_text}\n(Taille pinceau: {brush_size})', 
                            fontsize=12, fontweight='bold',
                            color='blue' if erase_mode else 'red')
            fig.canvas.draw_idle()
        elif event.key == 'c':
            mask.fill(0)
            update_display()
            print("🗑️ Masque effacé complètement")
        elif event.key == 's':
            finished = True
            print("💾 Sauvegarde demandée - fermez la fenêtre pour continuer")
            plt.close(fig)
    
    def on_close(event):
        nonlocal finished
        finished = True
        print("✅ Annotation terminée")
    
    # Connecter les gestionnaires d'événements
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Configuration initiale du titre
    mode_text = 'DESSINER'
    axes[2].set_title(f'Masque - Mode: {mode_text}\n(Taille pinceau: {brush_size})', 
                    fontsize=12, fontweight='bold', color='red')
    
    # Afficher l'interface
    plt.tight_layout()
    
    # Activer le mode interactif
    plt.ion()
    plt.show()
    
    # Attendre que l'utilisateur ferme la fenêtre
    print("\n⏳ Interface prête. Commencez à dessiner sur l'image de droite...")
    
    # Garder la fenêtre ouverte jusqu'à ce qu'elle soit fermée
    while plt.get_fignums() and fig.number in plt.get_fignums():
        plt.pause(0.1)
    
    print(f"✅ Masque final prêt pour sauvegarde (pixels non-zéro: {np.count_nonzero(mask)})")
    return mask

# Définir la fonction interactive_annotation comme alias pour la version simple
def interactive_annotation(image, initial_mask=None):
    """
    Fonction d'annotation interactive qui utilise l'interface matplotlib optimisée.
    """
    print("🎨 Lancement de l'interface d'annotation matplotlib...")
    return interactive_annotation_simple(image, initial_mask)

def save_results(image, mask, image_path, output_dir):
    """
    Sauvegarde les résultats de la détection.
    
    Args:
        image (numpy.ndarray): Image originale
        mask (numpy.ndarray): Masque des dislocations
        image_path (str): Chemin de l'image originale
        output_dir (str): Répertoire de sortie
    """
    # Extraire le nom de base de l'image
    base_name = os.path.basename(image_path).split('.')[0]
    
    # Définir les chemins de sortie
    image_out_path = os.path.join(output_dir, "images", f"{base_name}.tif")
    mask_out_path = os.path.join(output_dir, "masks", f"{base_name}_mask.png")
    
    # Sauvegarder l'image originale
    cv2.imwrite(image_out_path, image)
    
    # Sauvegarder le masque
    cv2.imwrite(mask_out_path, mask)
    
    print(f"Image sauvegardée sous: {image_out_path}")
    print(f"Masque sauvegardé sous: {mask_out_path}")
    
    # Créer une visualisation de la superposition
    overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = image  # Canal rouge
    overlay[:, :, 1] = image  # Canal vert
    overlay[:, :, 2] = image  # Canal bleu
    
    # Ajouter le masque en rouge
    overlay[mask > 0, 0] = 255  # Rouge pour les dislocations
    overlay[mask > 0, 1] = 0
    overlay[mask > 0, 2] = 0
    
    # Sauvegarder la superposition
    overlay_out_path = os.path.join(output_dir, "images", f"{base_name}_overlay.png")
    cv2.imwrite(overlay_out_path, overlay)
    print(f"Superposition sauvegardée sous: {overlay_out_path}")

def process_image(image_path, output_dir, method='adaptive', params=None, show_result=True, interactive=False):
    """
    Traite une image ECCI pour détecter les dislocations comme points.
    
    Args:
        image_path (str): Chemin vers l'image ECCI
        output_dir (str): Répertoire de sortie
        method (str): Méthode de détection ('adaptive', 'log', 'peak')
        params (dict): Paramètres de détection
        show_result (bool): Afficher le résultat
        interactive (bool): Activer l'annotation manuelle
        
    Returns:
        tuple: (image originale, masque des dislocations)
    """
    # Paramètres par défaut pour la méthode de seuillage adaptatif
    default_adaptive_params = {
        'adaptive_block_size': 51,    # Taille du bloc pour le seuillage adaptatif
        'adaptive_c': 5,              # Constante C pour le seuillage adaptatif
        'min_area': 5,                # Aire minimale des points (en pixels)
        'max_area': 100,              # Aire maximale des points (en pixels)
        'min_circularity': 0.5,       # Circularité minimale (0-1)
        'morph_kernel_size': 3,       # Taille du noyau pour les opérations morphologiques
        'gaussian_sigma': 1.0,        # Sigma pour le filtre gaussien
    }
    
    # Paramètres par défaut pour la méthode LoG (Laplacian of Gaussian)
    default_log_params = {
        'min_sigma': 1,               # Sigma minimum pour LoG
        'max_sigma': 5,               # Sigma maximum pour LoG
        'num_sigma': 10,              # Nombre de niveaux de sigma
        'threshold': 0.1,             # Seuil de détection
        'overlap': 0.5,               # Seuil de chevauchement
        'gaussian_sigma': 1.0,        # Sigma pour le filtre gaussien
        'morph_kernel_size': 3,       # Taille du noyau pour les opérations morphologiques
    }
    
    # Paramètres par défaut pour la méthode des maxima locaux
    default_peak_params = {
        'min_distance': 10,           # Distance minimale entre les pics
        'threshold_abs': 0.1,         # Seuil absolu
        'threshold_rel': 0.2,         # Seuil relatif
        'exclude_border': True,       # Exclure les bords
        'point_radius': 5,            # Rayon des points détectés
        'gaussian_sigma': 1.0,        # Sigma pour le filtre gaussien
        'morph_kernel_size': 3,       # Taille du noyau pour les opérations morphologiques
    }
    
    # Utiliser les paramètres fournis ou les paramètres par défaut
    if params is None:
        if method == 'log':
            params = default_log_params
        elif method == 'peak':
            params = default_peak_params
        else:  # méthode par défaut: adaptive
            params = default_adaptive_params
    
    # Charger l'image
    image = load_image(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return None, None
    
    # Prétraiter l'image
    preprocessed = preprocess_image(image, params['gaussian_sigma'])
    
    # Détecter les dislocations comme points selon la méthode choisie
    if method == 'log':
        mask, points = detect_dislocation_points_log(preprocessed, params)
    elif method == 'peak':
        mask, points = detect_dislocation_points_peak(preprocessed, params)
    else:  # méthode par défaut: adaptive
        mask, points = detect_dislocation_points(preprocessed, params)
    
    # Afficher les résultats si demandé
    if show_result:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Image originale
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Image originale')
        axes[0, 0].axis('off')
        
        # Image prétraitée
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Image prétraitée')
        axes[0, 1].axis('off')
        
        # Masque des dislocations
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Masque des dislocations')
        axes[1, 0].axis('off')
        
        # Superposition des points sur l'image originale
        axes[1, 1].imshow(image, cmap='gray')
        axes[1, 1].set_title(f'Points détectés ({len(points)})')
        axes[1, 1].axis('off')
        
        # Dessiner les cercles autour des points détectés
        for x, y, r in points:
            circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=1.5)
            axes[1, 1].add_patch(circle)
        
        plt.tight_layout()
        plt.show()
    
    # Annotation manuelle si demandée
    if interactive:
        print(f"Annotation manuelle pour {os.path.basename(image_path)}")
        mask = interactive_annotation(image, mask)
    
    return image, mask

def main():
    """
    Fonction principale pour exécuter le script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Détection de dislocations sous forme de points dans les images ECCI')
    parser.add_argument('--input', type=str, default=input_dir, 
                        help=f'Chemin vers l\'image ECCI ou le répertoire contenant les images (défaut: {input_dir})')
    parser.add_argument('--output', type=str, default=output_dir, 
                        help=f'Répertoire de sortie pour les résultats (défaut: {output_dir})')
    parser.add_argument('--method', type=str, choices=['adaptive', 'log', 'peak'], default='adaptive', 
                        help='Méthode de détection (défaut: adaptive)')
    parser.add_argument('--interactive', action='store_true', 
                        help='Activer l\'annotation manuelle')
    parser.add_argument('--no-show', action='store_true', 
                        help='Ne pas afficher les résultats')
    
    args = parser.parse_args()
    
    # Vérifier si l'entrée est un fichier ou un répertoire
    if os.path.isfile(args.input):
        # Traiter une seule image
        print(f"Traitement de l'image: {args.input}")
        image, mask = process_image(args.input, args.output, args.method, 
                                   show_result=not args.no_show, 
                                   interactive=args.interactive)
        
        if image is not None and mask is not None:
            # Sauvegarder les résultats
            save_results(image, mask, args.input, args.output)
    else:
        # Traiter un répertoire d'images
        image_paths = glob.glob(os.path.join(args.input, "*.tif"))
        
        if not image_paths:
            print(f"Aucune image TIFF trouvée dans {args.input}")
            return
        
        for image_path in image_paths:
            print(f"Traitement de l'image: {image_path}")
            image, mask = process_image(image_path, args.output, args.method, 
                                       show_result=not args.no_show, 
                                       interactive=args.interactive)
            
            if image is not None and mask is not None:
                # Sauvegarder les résultats
                save_results(image, mask, image_path, args.output)

if __name__ == "__main__":
    main()