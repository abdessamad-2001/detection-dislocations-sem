import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage import io, exposure, util
from skimage.feature import blob_log
import tkinter as tk
from tkinter import filedialog

plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['figure.dpi'] = 100

def load_image(image_path):
    try:
        image = io.imread(image_path)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path}: {e}")
        return None

def preprocess_image(image, gaussian_sigma=1.0):
    image_eq = exposure.equalize_hist(image)
    image_eq = util.img_as_ubyte(image_eq)
    image_filtered = cv2.GaussianBlur(image_eq, (0, 0), gaussian_sigma)
    return image_filtered

def detect_dislocation_points_log(image, params):
    mask = np.zeros_like(image)
    image_norm = image.astype(float) / 255.0
    image_inv = 1.0 - image_norm
    blobs = blob_log(image_inv,
                    min_sigma=params['min_sigma'],
                    max_sigma=params['max_sigma'],
                    num_sigma=params['num_sigma'],
                    threshold=params['threshold'],
                    overlap=params['overlap'])
    points = []
    for blob in blobs:
        y, x, sigma = blob
        radius = int(sigma * np.sqrt(2))
        points.append((int(x), int(y), radius))
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    return mask, points

def save_results(image, mask, points, output_dir, base_name):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "points"), exist_ok=True)

    image_out_path = os.path.join(output_dir, "images", f"{base_name}.tif")
    mask_out_path = os.path.join(output_dir, "masks", f"{base_name}_mask.png")
    points_out_path = os.path.join(output_dir, "points", f"{base_name}_points.csv")

    cv2.imwrite(image_out_path, image)
    cv2.imwrite(mask_out_path, mask)

    with open(points_out_path, 'w') as f:
        f.write("x,y,radius\n")
        for x, y, r in points:
            f.write(f"{x},{y},{r}\n")

    overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = image
    overlay[:, :, 1] = image
    overlay[:, :, 2] = image
    overlay[mask > 0, 0] = 255
    overlay[mask > 0, 1] = 0
    overlay[mask > 0, 2] = 0

    overlay_out_path = os.path.join(output_dir, "images", f"{base_name}_overlay.png")
    cv2.imwrite(overlay_out_path, overlay)
    print(f"Résultats sauvegardés dans {output_dir}.")

class InteractiveDetector:
    def __init__(self):
        self.log_params = {
            'min_sigma': 1,
            'max_sigma': 5,
            'num_sigma': 10,
            'threshold': 0.1,
            'overlap': 0.5,
            'gaussian_sigma': 1.0
        }
        self.image_path = None
        self.output_dir = None
        self.image = None
        self.preprocessed = None
        self.mask = None
        self.points = []
        self.method = 'log'

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)

        self.img_display = [None] * 4
        for i in range(4):
            self.img_display[i] = self.axes[i // 2, i % 2].imshow(np.zeros((100, 100)), cmap='gray')
            self.axes[i // 2, i % 2].axis('off')

        self.axes[0, 0].set_title('Image originale')
        self.axes[0, 1].set_title('Image prétraitée')
        self.axes[1, 0].set_title('Masque des dislocations')
        self.axes[1, 1].set_title('Points détectés (0)')

        self.load_button_ax = plt.axes([0.3, 0.25, 0.15, 0.05])
        self.load_button = Button(self.load_button_ax, 'Charger une image')
        self.load_button.on_clicked(self.load_image_dialog)

        self.output_button_ax = plt.axes([0.5, 0.25, 0.15, 0.05])
        self.output_button = Button(self.output_button_ax, 'Choisir dossier de sortie')
        self.output_button.on_clicked(self.select_output_dir)

        self.save_button_ax = plt.axes([0.7, 0.25, 0.15, 0.05])
        self.save_button = Button(self.save_button_ax, 'Sauvegarder les résultats')
        self.save_button.on_clicked(self.save_current_results)

        self.annotate_button_ax = plt.axes([0.1, 0.01, 0.2, 0.05])
        self.annotate_button = Button(self.annotate_button_ax, 'Annotation manuelle')
        self.annotate_button.on_clicked(self.run_manual_annotation)

        self.sliders = {}
        self.create_log_sliders()
        plt.show()

    def create_log_sliders(self):
        for slider in self.sliders.values():
            slider.ax.remove()
        self.sliders = {}

        ax_min_sigma = plt.axes([0.1, 0.2, 0.8, 0.02])
        self.sliders['min_sigma'] = Slider(ax_min_sigma, 'Sigma min', 0.5, 5.0, valinit=self.log_params['min_sigma'])
        self.sliders['min_sigma'].on_changed(self.update_log_params)

        ax_max_sigma = plt.axes([0.1, 0.17, 0.8, 0.02])
        self.sliders['max_sigma'] = Slider(ax_max_sigma, 'Sigma max', 1.0, 10.0, valinit=self.log_params['max_sigma'])
        self.sliders['max_sigma'].on_changed(self.update_log_params)

        ax_num_sigma = plt.axes([0.1, 0.14, 0.8, 0.02])
        self.sliders['num_sigma'] = Slider(ax_num_sigma, 'Nb sigmas', 5, 20, valinit=self.log_params['num_sigma'], valstep=1)
        self.sliders['num_sigma'].on_changed(self.update_log_params)

        ax_threshold = plt.axes([0.1, 0.11, 0.8, 0.02])
        self.sliders['threshold'] = Slider(ax_threshold, 'Seuil', 0.01, 0.5, valinit=self.log_params['threshold'])
        self.sliders['threshold'].on_changed(self.update_log_params)

        ax_overlap = plt.axes([0.1, 0.08, 0.8, 0.02])
        self.sliders['overlap'] = Slider(ax_overlap, 'Chevauchement', 0.1, 1.0, valinit=self.log_params['overlap'])
        self.sliders['overlap'].on_changed(self.update_log_params)

        ax_sigma = plt.axes([0.1, 0.05, 0.8, 0.02])
        self.sliders['gaussian_sigma'] = Slider(ax_sigma, 'Sigma gaussien', 0.1, 5.0, valinit=self.log_params['gaussian_sigma'])
        self.sliders['gaussian_sigma'].on_changed(self.update_log_params)

        plt.draw()

    def update_log_params(self, val):
        self.log_params['min_sigma'] = self.sliders['min_sigma'].val
        self.log_params['max_sigma'] = self.sliders['max_sigma'].val
        self.log_params['num_sigma'] = int(self.sliders['num_sigma'].val)
        self.log_params['threshold'] = self.sliders['threshold'].val
        self.log_params['overlap'] = self.sliders['overlap'].val
        self.log_params['gaussian_sigma'] = self.sliders['gaussian_sigma'].val
        if self.image is not None:
            self.update_detection()

    def load_image_dialog(self, event):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.tif *.jpg *.png")])
        if file_path:
            self.image_path = file_path
            self.load_image()

    def load_image(self):
        if self.image_path:
            self.image = load_image(self.image_path)
            for i in range(4):
                self.img_display[i].remove()
            self.img_display[0] = self.axes[0, 0].imshow(self.image, cmap='gray', vmin=0, vmax=255)
            self.img_display[1] = self.axes[0, 1].imshow(np.zeros_like(self.image), cmap='gray', vmin=0, vmax=255)
            self.img_display[2] = self.axes[1, 0].imshow(np.zeros_like(self.image), cmap='gray', vmin=0, vmax=255)
            self.img_display[3] = self.axes[1, 1].imshow(np.zeros_like(self.image), cmap='gray', vmin=0, vmax=255)
            for ax in self.axes.flat:
                ax.set_xlim(0, self.image.shape[1])
                ax.set_ylim(self.image.shape[0], 0)
            self.update_detection()
            self.fig.canvas.draw_idle()

    def update_detection(self):
        if self.image is None:
            return
        self.preprocessed = preprocess_image(self.image, self.log_params['gaussian_sigma'])
        self.img_display[1].set_data(self.preprocessed)
        self.mask, self.points = detect_dislocation_points_log(self.preprocessed, self.log_params)
        self.img_display[2].set_data(self.mask)
        for artist in self.axes[1, 1].patches:
            artist.remove()
        self.img_display[3].set_data(self.image)
        self.axes[1, 1].set_title(f'Points détectés ({len(self.points)})')
        for x, y, r in self.points:
            circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=1.5)
            self.axes[1, 1].add_patch(circle)
        self.fig.canvas.draw_idle()

    def select_output_dir(self, event):
        root = tk.Tk()
        root.withdraw()
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir = dir_path
            print(f"Répertoire de sortie sélectionné: {self.output_dir}")

    def save_current_results(self, event):
        if self.image is None or self.output_dir is None:
            print("Erreur: image ou dossier non spécifié")
            return
        base_name = os.path.basename(self.image_path).split('.')[0]
        save_results(self.image, self.mask, self.points, self.output_dir, base_name)

    def run_manual_annotation(self, event):
        if self.image is None:
            print("Erreur : aucune image chargée.")
            return
        try:
            from creation_masques_ecci_Points import interactive_annotation_simple
        except ImportError:
            print("❌ Impossible d’importer la fonction d’annotation interactive.")
            return

        print("✏️ Lancement de l’annotation manuelle…")
        plt.ioff()
        mask_new = interactive_annotation_simple(self.image, self.mask)
        plt.ion()
        if mask_new is not None:
            self.mask = mask_new
            self.img_display[2].set_data(self.mask)
            self.img_display[2].set_clim(vmin=0, vmax=255)
            self.axes[1, 0].set_title('Masque des dislocations (modifié)')
            self.fig.canvas.draw_idle()
        else:
            print("Annotation annulée.")

def main():
    detector = InteractiveDetector()

if __name__ == "__main__":
    main()
