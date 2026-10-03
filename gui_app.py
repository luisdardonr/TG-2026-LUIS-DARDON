"""
Interfaz gráfica principal del sistema de reconocimiento facial.

Estructura de la ventana:
    - Encabezado con el título del proyecto.
    - Barra lateral de navegación (una sección por función: reconocer,
      comparar, gestionar personas/dataset, entrenar, evaluar).
    - Panel central: contenido de la sección activa.
    - Panel derecho: registro de actividad, siempre visible.

Toda la lógica de reconocimiento, entrenamiento y persistencia vive en
los módulos recognition_engine / classifier / face_store / metrics;
este archivo solo se encarga de la interfaz y de conectar los botones
con esas funciones.
"""

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import insightface
import numpy as np
from PIL import Image, ImageTk

from classifier import load_classifier, train_classifier_from_samples
from config import (
    AUTO_LABEL,
    DATASET_PATH,
    DEFAULT_COSINE_THRESHOLD,
    EVAL_LOG_INTERVAL,
    METRICS_LOG_PATH,
    MIN_SAMPLES_PER_PERSON,
)
from dataset_eval import evaluate_dataset
from face_store import (
    extract_embeddings_from_video,
    import_dataset_from_path,
    load_known_faces,
    save_known_faces,
)
from geometry import extract_geometric_features
from metrics import ensure_metrics_file, log_metric
from recognition_engine import RecognitionEngine
from theme import (
    COLOR_ACCENT,
    COLOR_CARD,
    COLOR_MUTED,
    COLOR_SIDEBAR,
    COLOR_SIDEBAR_TEXT,
    COLOR_SUCCESS,
    apply_theme,
)

SECTIONS = [
    ("reconocimiento", "Reconocimiento en vivo"),
    ("comparar", "Comparar imágenes"),
    ("personas", "Personas y dataset"),
    ("entrenamiento", "Entrenamiento"),
    ("evaluacion", "Evaluación"),
]


class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial - Tesis UVG")
        self.root.geometry("1360x820")
        self.root.minsize(1120, 680)
        apply_theme(self.root)

        # Motor de reconocimiento (encapsula known_faces + clasificador)
        self.engine = RecognitionEngine()
        self.engine.threshold = DEFAULT_COSINE_THRESHOLD

        # Modelo insightface (backbone preentrenado, congelado)
        self.face_app = insightface.app.FaceAnalysis()
        self.face_app.prepare(ctx_id=-1)  # -1 = CPU, 0 = GPU si disponible

        # Estado de imagen / video
        self.img1_path = None
        self.img2_path = None
        self.video_running = False
        self.video_capture = None
        self.active_section = "reconocimiento"

        # Estado de la sesión de evaluación de métricas
        self.eval_active = False
        self.eval_ground_truth = None
        self.eval_counts = {"correct": 0, "total": 0}
        self.eval_frame_counter = 0
        ensure_metrics_file()

        self.nav_buttons = {}
        self.sections = {}

        self._build_layout()
        self._load_initial_data()
        self.show_section("reconocimiento")

    # ==================================================================
    # Construcción de la interfaz
    # ==================================================================

    def _build_layout(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self._build_header()

        self.sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=240)
        self.sidebar.grid(row=1, column=0, sticky="nswe")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        self.content = ttk.Frame(self.root, style="TFrame")
        self.content.grid(row=1, column=1, sticky="nswe", padx=(16, 8), pady=16)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        # Antes ocupaba 300px fijos siempre. Ahora arranca más angosto
        # (220px) y además se puede plegar a una franja delgada con el
        # botón "—" del encabezado del panel (ver toggle_results_panel).
        self.results_panel = ttk.Frame(self.root, style="Card.TFrame", width=220)
        self.results_panel.grid(row=1, column=2, sticky="nswe", padx=(8, 16), pady=16)
        self.results_panel.grid_propagate(False)
        self._build_results_panel()

        self._build_section_reconocimiento()
        self._build_section_comparar()
        self._build_section_personas()
        self._build_section_entrenamiento()
        self._build_section_evaluacion()

    def _build_header(self):
        header = ttk.Frame(self.root, style="TFrame")
        header.grid(row=0, column=0, columnspan=3, sticky="we")

        inner = ttk.Frame(header, style="TFrame")
        inner.pack(fill="x", padx=20, pady=(16, 10))
        ttk.Label(inner, text="Sistema de Reconocimiento Facial Adaptable",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(inner, text="Trabajo de graduación · Ingeniería Mecatrónica · UVG",
                  style="MutedOnBg.TLabel").pack(anchor="w")

        ttk.Separator(header, orient="horizontal").pack(fill="x")

    def _build_sidebar(self):
        for i, (key, label) in enumerate(SECTIONS):
            btn = tk.Button(
                self.sidebar, text=label, anchor="w", bd=0, padx=18, pady=13,
                font=("Segoe UI", 11), bg=COLOR_SIDEBAR, fg=COLOR_SIDEBAR_TEXT,
                activebackground=COLOR_ACCENT, activeforeground="white",
                relief="flat", cursor="hand2",
                command=lambda k=key: self.show_section(k),
            )
            btn.pack(fill="x", pady=(20 if i == 0 else 1, 0), padx=10)
            self.nav_buttons[key] = btn

        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=18, padx=10)

        info_frame = tk.Frame(self.sidebar, bg=COLOR_SIDEBAR)
        info_frame.pack(fill="x", padx=18)

        tk.Label(info_frame, text="ESTADO DEL SISTEMA", bg=COLOR_SIDEBAR, fg=COLOR_MUTED,
                  font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 6))

        self.classifier_status_label = tk.Label(
            info_frame, text="Clasificador: sin entrenar", bg=COLOR_SIDEBAR,
            fg=COLOR_MUTED, font=("Segoe UI", 9), justify="left", wraplength=200)
        self.classifier_status_label.pack(anchor="w", pady=(0, 6))

        self.people_count_label = tk.Label(
            info_frame, text="0 personas registradas", bg=COLOR_SIDEBAR,
            fg=COLOR_MUTED, font=("Segoe UI", 9))
        self.people_count_label.pack(anchor="w")

    def _build_results_panel(self):
        header = tk.Frame(self.results_panel, bg=COLOR_CARD)
        header.pack(fill="x", padx=10, pady=(10, 4))

        tk.Label(header, text="Registro", bg=COLOR_CARD, fg="#1c2434",
                  font=("Segoe UI", 10, "bold")).pack(side="left")

        self.results_toggle_btn = tk.Button(
            header, text="—", bd=0, bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 10, "bold"),
            cursor="hand2", relief="flat", width=2, command=self.toggle_results_panel,
        )
        self.results_toggle_btn.pack(side="right")

        tk.Button(
            header, text="🗑", bd=0, bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 10),
            cursor="hand2", relief="flat", width=2, command=self.clear_results_panel,
        ).pack(side="right")

        self.results_body = tk.Frame(self.results_panel, bg=COLOR_CARD)
        self.results_body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Texto compacto (fuente más pequeña que el resto de la
        # interfaz) porque este panel es un registro secundario, no el
        # contenido principal de la pantalla.
        self.result_text = tk.Text(self.results_body, wrap="word", font=("Consolas", 8),
                                    bg="#f8fafc", relief="flat", bd=0, padx=8, pady=8)
        scrollbar = ttk.Scrollbar(self.results_body, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.results_collapsed = False

    def toggle_results_panel(self):
        """Pliega el panel a una franja delgada (36px) o lo vuelve a
        expandir a su ancho normal (220px)."""
        if self.results_collapsed:
            self.results_panel.configure(width=220)
            self.results_body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.results_toggle_btn.config(text="—")
        else:
            self.results_panel.configure(width=34)
            self.results_body.pack_forget()
            self.results_toggle_btn.config(text="▤")
        self.results_collapsed = not self.results_collapsed

    def clear_results_panel(self):
        self.result_text.delete("1.0", tk.END)

    # ------------------------------------------------------------------
    # Sección: Reconocimiento en vivo
    # ------------------------------------------------------------------
    def _build_section_reconocimiento(self):
        frame = ttk.Frame(self.content, style="Card.TFrame")
        self.sections["reconocimiento"] = frame
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        top = ttk.Frame(frame, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="we", padx=24, pady=(24, 12))

        ttk.Label(top, text="Reconocimiento en vivo", style="SectionTitle.TLabel").grid(
            row=0, column=0, columnspan=5, sticky="w", pady=(0, 14))

        ttk.Label(top, text="Persona de referencia:", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.reference_person = ttk.Combobox(top, values=self.get_reference_options(), width=26, state="readonly")
        self.reference_person.set(AUTO_LABEL)
        self.reference_person.grid(row=1, column=1, padx=(8, 24), sticky="w")

        ttk.Label(top, text="Umbral coseno:", style="Card.TLabel").grid(row=1, column=2, sticky="w")
        self.threshold_var = tk.DoubleVar(value=self.engine.threshold)
        self.threshold_slider = ttk.Scale(top, from_=0.3, to=1.0, variable=self.threshold_var,
                                           orient="horizontal", command=self.update_threshold, length=150)
        self.threshold_slider.grid(row=1, column=3, sticky="w", padx=(8, 0))
        self.threshold_label = ttk.Label(top, text=f"{self.engine.threshold:.2f}", style="Card.TLabel")
        self.threshold_label.grid(row=1, column=4, sticky="w", padx=(8, 0))

        ttk.Button(top, text="Registrar nuevo rostro", style="Accent.TButton",
                   command=self.register_new_face).grid(row=2, column=0, pady=(16, 0), sticky="w")

        self.camera_toggle_btn = ttk.Button(top, text="Apagar cámara", command=self.toggle_camera)
        self.camera_toggle_btn.grid(row=2, column=1, pady=(16, 0), padx=(10, 0), sticky="w")

        self.camera_status_label = ttk.Label(top, text="", style="Muted.TLabel")
        self.camera_status_label.grid(row=2, column=2, columnspan=2, pady=(16, 0), padx=(10, 0), sticky="w")

        video_frame = ttk.Frame(frame, style="TFrame")
        video_frame.grid(row=1, column=0, sticky="nswe", padx=24, pady=(0, 24))
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Sección: Comparar imágenes
    # ------------------------------------------------------------------
    def _build_section_comparar(self):
        frame = ttk.Frame(self.content, style="Card.TFrame")
        self.sections["comparar"] = frame
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ttk.Label(frame, text="Comparar dos imágenes (verificación 1 a 1)",
                  style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w", padx=24, pady=(24, 14))

        controls = ttk.Frame(frame, style="Card.TFrame")
        controls.grid(row=1, column=0, sticky="we", padx=24)

        ttk.Label(controls, text="Imagen 1 (Referencia):", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.img1_label = ttk.Label(controls, text="Ninguna imagen seleccionada", style="Muted.TLabel")
        self.img1_label.grid(row=0, column=1, padx=12, sticky="w")
        ttk.Button(controls, text="Cargar", command=self.load_image1).grid(row=0, column=2, padx=6)

        ttk.Label(controls, text="Imagen 2 (Comparar):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        self.img2_label = ttk.Label(controls, text="Ninguna imagen seleccionada", style="Muted.TLabel")
        self.img2_label.grid(row=1, column=1, padx=12, sticky="w", pady=(10, 0))
        ttk.Button(controls, text="Cargar", command=self.load_image2).grid(row=1, column=2, padx=6, pady=(10, 0))

        ttk.Button(controls, text="Comparar imágenes", style="Accent.TButton",
                   command=self.compare_images).grid(row=2, column=0, columnspan=3, pady=18, sticky="w")

        preview_frame = ttk.Frame(frame, style="TFrame")
        preview_frame.grid(row=2, column=0, sticky="nswe", padx=24, pady=(0, 24))
        self.compare_label = tk.Label(preview_frame, bg="black")
        self.compare_label.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Sección: Personas y dataset
    # ------------------------------------------------------------------
    def _build_section_personas(self):
        frame = ttk.Frame(self.content, style="Card.TFrame")
        self.sections["personas"] = frame
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ttk.Label(frame, text="Personas registradas y datos", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=24, pady=(24, 14))

        list_frame = ttk.Frame(frame, style="Card.TFrame")
        list_frame.grid(row=1, column=0, sticky="nswe", padx=24)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        columns = ("nombre", "capturas")
        self.people_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        self.people_tree.heading("nombre", text="Nombre")
        self.people_tree.heading("capturas", text="Capturas registradas")
        self.people_tree.column("nombre", width=260)
        self.people_tree.column("capturas", width=160, anchor="center")
        self.people_tree.grid(row=0, column=0, sticky="nswe")

        tree_scroll = ttk.Scrollbar(list_frame, command=self.people_tree.yview)
        self.people_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky="ns")

        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="we", padx=24, pady=18)
        ttk.Button(actions, text="Importar dataset (carpeta)",
                   command=self.import_dataset_folder).pack(side="left", padx=(0, 10))
        ttk.Button(actions, text="Importar desde video",
                   command=self.import_from_video).pack(side="left", padx=(0, 10))
        ttk.Button(actions, text="Eliminar persona seleccionada", style="Danger.TButton",
                   command=self.delete_selected_person).pack(side="left")

    # ------------------------------------------------------------------
    # Sección: Entrenamiento
    # ------------------------------------------------------------------
    def _build_section_entrenamiento(self):
        frame = ttk.Frame(self.content, style="Card.TFrame")
        self.sections["entrenamiento"] = frame

        ttk.Label(frame, text="Entrenamiento del clasificador", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=24, pady=(24, 14))

        ttk.Label(
            frame,
            text="Red neuronal feedforward (256 → 128 → clases) entrenada con\n"
                 "embeddings ArcFace + características geométricas. Se recomienda\n"
                 f"un mínimo de {MIN_SAMPLES_PER_PERSON} capturas por persona antes de entrenar.",
            style="Card.TLabel", justify="left",
        ).grid(row=1, column=0, sticky="w", padx=24)

        self.classifier_detail_label = ttk.Label(frame, text="Estado: sin entrenar", style="Muted.TLabel")
        self.classifier_detail_label.grid(row=2, column=0, sticky="w", padx=24, pady=(16, 0))

        ttk.Button(frame, text="Entrenar modelo con datos actuales", style="Accent.TButton",
                   command=self.train_classifier).grid(row=3, column=0, sticky="w", padx=24, pady=20)

    # ------------------------------------------------------------------
    # Sección: Evaluación
    # ------------------------------------------------------------------
    def _build_section_evaluacion(self):
        frame = ttk.Frame(self.content, style="Card.TFrame")
        self.sections["evaluacion"] = frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(2, weight=1)

        # -- Columna izquierda: evaluación EN VIVO (cámara) -----------
        live = ttk.Frame(frame, style="Card.TFrame")
        live.grid(row=0, column=0, sticky="nwe", padx=(24, 12), pady=24)

        ttk.Label(live, text="Evaluación en vivo (cámara)", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 14))

        ttk.Label(live, text="Persona real frente a la cámara:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w")
        self.eval_person = ttk.Combobox(live, values=list(self.engine.known_faces.keys()),
                                         width=28, state="readonly")
        self.eval_person.grid(row=2, column=0, sticky="w", pady=(6, 12))

        self.eval_button = ttk.Button(live, text="Iniciar evaluación", style="Accent.TButton",
                                       command=self.toggle_evaluation)
        self.eval_button.grid(row=3, column=0, sticky="w")

        self.eval_status_label = ttk.Label(live, text="Evaluación: detenida", style="Muted.TLabel")
        self.eval_status_label.grid(row=4, column=0, sticky="w", pady=(12, 0))

        ttk.Label(
            live,
            text="Si la cámara está apagada, se enciende automáticamente\n"
                 "al iniciar la evaluación.",
            style="Muted.TLabel", justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(16, 0))

        ttk.Separator(frame, orient="vertical").grid(row=0, column=1, sticky="ns", pady=24)

        # -- Columna derecha: evaluación por LOTE sobre dataset -------
        batch = ttk.Frame(frame, style="Card.TFrame")
        batch.grid(row=0, column=2, sticky="nwe", padx=(12, 24), pady=24)

        ttk.Label(batch, text="Evaluación por lote (dataset)", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 14))

        ttk.Label(
            batch,
            text="Recorre una carpeta con estructura dataset/persona/*.jpg,\n"
                 "reconoce cada imagen con el motor actual (clasificador +\n"
                 "verificación por coseno) y compara contra el nombre de la\n"
                 "carpeta como identidad real. No requiere cámara ni depende\n"
                 "de estar frente al equipo -- ideal para correr sobre un\n"
                 "conjunto de prueba completo y obtener métricas reproducibles.\n\n"
                 "Además de precisión, calcula pérdida (cross-entropy), tiempo\n"
                 "de inferencia, calibración de confianza, y un análisis de\n"
                 "robustez ante oclusión (cejas/ojos) para casos como el uso\n"
                 "de mascarilla. Detalle por imagen exportado a CSV.",
            style="Card.TLabel", justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 14))

        ttk.Button(batch, text="Seleccionar carpeta y evaluar", style="Accent.TButton",
                   command=self.run_dataset_evaluation).grid(row=2, column=0, sticky="w")

        self.dataset_eval_status_label = ttk.Label(batch, text="Sin evaluar todavía.",
                                                     style="Muted.TLabel", justify="left")
        self.dataset_eval_status_label.grid(row=3, column=0, sticky="w", pady=(14, 0))

        ttk.Label(
            frame,
            text=f"Ambos tipos de evaluación (en vivo y por dataset) se guardan en "
                 f"'{METRICS_LOG_PATH}', distinguibles por la columna 'fuente'.",
            style="Muted.TLabel", justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=24, pady=(0, 20))

    # ==================================================================
    # Navegación
    # ==================================================================

    def show_section(self, key):
        self.active_section = key
        for frame in self.sections.values():
            frame.grid_remove()
        self.sections[key].grid(row=0, column=0, sticky="nswe")

        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(bg=COLOR_ACCENT, fg="white")
            else:
                btn.configure(bg=COLOR_SIDEBAR, fg=COLOR_SIDEBAR_TEXT)

        if key == "comparar":
            self.show_image_placeholder()

        # La cámara ya NO se mantiene encendida todo el tiempo (eso
        # consumía CPU en segundo plano incluso viendo otras
        # secciones). Se enciende solo al entrar a "Reconocimiento", y
        # se apaga al salir -- excepto si hay una evaluación en curso,
        # que la necesita activa para seguir midiendo.
        if key == "reconocimiento":
            self.start_video()
        elif not self.eval_active:
            self.stop_video()

        self._refresh_camera_controls()

    # ==================================================================
    # Carga inicial de datos
    # ==================================================================

    def _load_initial_data(self):
        known_faces, messages = load_known_faces()
        self.engine.known_faces = known_faces
        for m in messages:
            self.update_result_text(m + "\n")

        if not self.engine.known_faces and Path(DATASET_PATH).exists():
            self.update_result_text(
                f"No hay personas registradas. Importando dataset por defecto desde '{DATASET_PATH}'...\n"
            )
            resumen, total = import_dataset_from_path(self.engine.known_faces, DATASET_PATH, self.face_app)
            if total:
                self.update_result_text(
                    "\nDataset importado:\n" + "\n".join(resumen) +
                    "\n(Recuerda entrenar el modelo para que use estos datos)\n"
                )

        if len(self.engine.known_faces) == 0:
            self.update_result_text(
                "No hay rostros registrados. Use 'Registrar nuevo rostro' o importe un dataset/video.\n"
            )

        model, labels, fmean, fstd, warning = load_classifier()
        if warning:
            self.update_result_text(warning + "\n")
        elif model is not None:
            self.engine.classifier = model
            self.engine.class_labels = labels
            self.engine.feature_mean = fmean
            self.engine.feature_std = fstd
            self.update_result_text(f"Clasificador cargado ({len(labels)} personas)\n")

        self.refresh_person_lists()
        self.refresh_classifier_status()

        # La cámara ya no se enciende aquí de forma incondicional: se
        # enciende/apaga según la sección activa (ver show_section),
        # para no consumir CPU en segundo plano cuando no se está
        # usando el video en vivo.

    # ==================================================================
    # Auxiliares de estado / listas
    # ==================================================================

    def get_reference_options(self):
        return [AUTO_LABEL] + list(self.engine.known_faces.keys())

    def refresh_person_lists(self):
        if hasattr(self, "reference_person"):
            self.reference_person["values"] = self.get_reference_options()
        if hasattr(self, "eval_person"):
            self.eval_person["values"] = list(self.engine.known_faces.keys())
        if hasattr(self, "people_tree"):
            self.people_tree.delete(*self.people_tree.get_children())
            for name, samples in sorted(self.engine.known_faces.items()):
                self.people_tree.insert("", "end", values=(name, len(samples)))
        total_people = len(self.engine.known_faces)
        if hasattr(self, "people_count_label"):
            self.people_count_label.config(text=f"{total_people} persona(s) registrada(s)")

    def refresh_classifier_status(self):
        if self.engine.classifier is not None:
            self.classifier_status_label.config(
                text=f"Clasificador: entrenado\n({len(self.engine.class_labels)} personas)",
                fg=COLOR_SUCCESS,
            )
            if hasattr(self, "classifier_detail_label"):
                self.classifier_detail_label.config(
                    text=f"Estado: entrenado con {len(self.engine.class_labels)} personas",
                    foreground=COLOR_SUCCESS,
                )
        else:
            self.classifier_status_label.config(text="Clasificador: sin entrenar", fg=COLOR_MUTED)
            if hasattr(self, "classifier_detail_label"):
                self.classifier_detail_label.config(text="Estado: sin entrenar", foreground=COLOR_MUTED)

    def update_threshold(self, value):
        self.engine.threshold = float(value)
        self.threshold_label.config(text=f"{self.engine.threshold:.2f}")

    def update_result_text(self, text):
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)

    # ==================================================================
    # Modo comparación de imágenes
    # ==================================================================

    def load_image1(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen de referencia",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")],
        )
        if filepath:
            self.img1_path = filepath
            self.img1_label.config(text=os.path.basename(filepath), foreground=COLOR_SUCCESS)
            self.update_result_text(f"Imagen 1 cargada: {os.path.basename(filepath)}\n")

    def load_image2(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen a comparar",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")],
        )
        if filepath:
            self.img2_path = filepath
            self.img2_label.config(text=os.path.basename(filepath), foreground=COLOR_SUCCESS)
            self.update_result_text(f"Imagen 2 cargada: {os.path.basename(filepath)}\n")

    def compare_images(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Advertencia", "Debe cargar ambas imágenes")
            return

        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        if img1 is None or img2 is None:
            messagebox.showerror("Error", "No se pudieron cargar las imágenes")
            return

        faces1 = self.face_app.get(img1)
        faces2 = self.face_app.get(img2)
        if len(faces1) == 0:
            self.update_result_text("No se detectó rostro en Imagen 1\n")
            return
        if len(faces2) == 0:
            self.update_result_text("No se detectó rostro en Imagen 2\n")
            return

        embedding1 = faces1[0].embedding
        embedding2 = faces2[0].embedding
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        percentage = similarity * 100

        self.update_result_text("\n" + "=" * 40 + "\n")
        self.update_result_text("RESULTADO DE COMPARACIÓN:\n")
        self.update_result_text(f"   Similitud: {percentage:.2f}%\n")
        self.update_result_text(f"   Umbral usado: {self.engine.threshold * 100:.0f}%\n")
        if similarity > self.engine.threshold:
            self.update_result_text("   VEREDICTO: Misma persona\n")
        else:
            self.update_result_text("   VEREDICTO: Personas diferentes\n")
        self.update_result_text("=" * 40 + "\n")

        self.display_comparison_images(img1, img2, faces1[0].bbox, faces2[0].bbox, percentage)

    def display_comparison_images(self, img1, img2, bbox1, bbox2, percentage):
        img1_copy = img1.copy()
        img2_copy = img2.copy()

        x1, y1, x2, y2 = bbox1.astype(int)
        cv2.rectangle(img1_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img1_copy, "Referencia", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        x1, y1, x2, y2 = bbox2.astype(int)
        color = (0, 255, 0) if percentage / 100 > self.engine.threshold else (0, 0, 255)
        cv2.rectangle(img2_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img2_copy, f"Sim: {percentage:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ANTES: cada imagen se pegaba a su tamaño ORIGINAL (arriba a
        # la izquierda de su mitad del lienzo) y luego se forzaba TODO
        # el collage a 800x400. Si las dos fotos tenían proporciones
        # o resoluciones distintas (una vertical y angosta, otra
        # horizontal y ancha, por ejemplo), terminaban en escalas muy
        # distintas dentro del collage y el estiramiento final a
        # 800x400 las deformaba de forma distinta a cada una -- de ahí
        # el desfase/desalineación entre ambas.
        #
        # AHORA: primero se normalizan las dos a la MISMA altura
        # (conservando su proporción original), y solo si el collage
        # resultante es más ancho de lo razonable se reduce completo
        # manteniendo proporción -- nunca se deforma de forma distinta
        # cada mitad.
        target_height = 420

        def resize_to_height(img, height):
            h, w = img.shape[:2]
            scale = height / h
            new_w = max(1, int(round(w * scale)))
            return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)

        img1_resized = resize_to_height(img1_copy, target_height)
        img2_resized = resize_to_height(img2_copy, target_height)

        img1_rgb = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB)

        gap = 16
        w1 = img1_rgb.shape[1]
        w2 = img2_rgb.shape[1]

        combined = np.ones((target_height, w1 + gap + w2, 3), dtype=np.uint8) * 255
        combined[:, :w1] = img1_rgb
        combined[:, w1 + gap:w1 + gap + w2] = img2_rgb
        combined = cv2.putText(combined, f"Similitud: {percentage:.1f}%", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 0), 2)

        img_pil = Image.fromarray(combined)

        max_width = 900
        if img_pil.width > max_width:
            ratio = max_width / img_pil.width
            img_pil = img_pil.resize(
                (max_width, max(1, int(img_pil.height * ratio))), Image.Resampling.LANCZOS
            )

        imgtk = ImageTk.PhotoImage(img_pil)
        self.compare_label.config(image=imgtk)
        self.compare_label.image = imgtk

    def show_image_placeholder(self):
        placeholder = np.ones((360, 640, 3), dtype=np.uint8) * 40
        cv2.putText(placeholder, "Modo comparacion de imagenes", (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(placeholder, "Cargue las imagenes y presione Comparar", (40, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        img_pil = Image.fromarray(cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(img_pil)
        self.compare_label.config(image=imgtk)
        self.compare_label.image = imgtk

    # ==================================================================
    # Modo video en vivo
    # ==================================================================

    def start_video(self):
        if self.video_running:
            return
        self.video_running = True
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            self.video_running = False
            self._refresh_camera_controls()
            return
        self.update_video()
        self._refresh_camera_controls()

    def stop_video(self):
        self.video_running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        # Mientras la cámara está apagada no hay imagen que mostrar;
        # se deja un fondo negro simple en vez de un último frame
        # congelado, para que sea obvio que no está transmitiendo.
        if hasattr(self, "video_label"):
            self.video_label.config(image="")
            self.video_label.image = None
        self._refresh_camera_controls()

    def toggle_camera(self):
        """Botón manual en la sección Reconocimiento: permite apagar
        la cámara aunque se esté viendo esa sección (por ejemplo, para
        liberar recursos mientras se piensa qué hacer) y volver a
        encenderla cuando se necesite."""
        if self.video_running:
            self.stop_video()
            self.update_result_text("Cámara apagada manualmente.\n")
        else:
            self.start_video()
            self.update_result_text("Cámara encendida.\n")

    def _refresh_camera_controls(self):
        if not hasattr(self, "camera_toggle_btn"):
            return
        if self.video_running:
            self.camera_toggle_btn.config(text="Apagar cámara")
            self.camera_status_label.config(text="Cámara: encendida")
        else:
            self.camera_toggle_btn.config(text="Encender cámara")
            self.camera_status_label.config(text="Cámara: apagada")

    def update_video(self):
        if not self.video_running:
            return

        success, img = self.video_capture.read()
        if not success or img is None:
            self.root.after(10, self.update_video)
            return

        selected = self.reference_person.get() if hasattr(self, "reference_person") else AUTO_LABEL
        auto_mode = (selected == AUTO_LABEL or selected == "")
        only_person = None if auto_mode else selected

        faces = self.face_app.get(img)

        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            name, score = self.engine.recognize(face, only_person=only_person)
            percentage = score * 100

            if name:
                text = f"{name} {percentage:.1f}%"
                color = (0, 255, 0)
            else:
                text = f"Desconocido {percentage:.1f}%"
                color = (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if self.eval_active and i == 0:
                self.eval_frame_counter += 1
                if self.eval_frame_counter % EVAL_LOG_INTERVAL == 0:
                    eval_name, eval_score = self.engine.recognize(face, only_person=None)
                    is_correct = (eval_name == self.eval_ground_truth)
                    self.eval_counts["total"] += 1
                    if is_correct:
                        self.eval_counts["correct"] += 1
                    log_metric(self.eval_ground_truth, eval_name, eval_score, is_correct, source="vivo")
                    total = self.eval_counts["total"]
                    correct = self.eval_counts["correct"]
                    acc = (correct / total * 100) if total else 0
                    self.eval_status_label.config(
                        text=f"Evaluando: {self.eval_ground_truth} ({correct}/{total}, {acc:.0f}%)"
                    )

        # El clasificador ahora también se usa en verificación 1 a 1
        # cuando la persona seleccionada es una de sus clases
        # conocidas (ver RecognitionEngine.recognize) -- así que el
        # letrero debe reflejar eso en vez de asumir que solo el modo
        # automático lo usa.
        classifier_usable = self.engine.classifier is not None and (
            auto_mode or selected in self.engine.class_labels
        )
        motor = "clasificador entrenado" if classifier_usable else "similitud (coseno)"
        if auto_mode:
            modo_texto = f"Automático ({motor})"
        else:
            modo_texto = f"Verificación 1 a 1: {selected} ({motor})"
        cv2.putText(img, f"Modo: {modo_texto}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Umbral coseno: {self.engine.threshold:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(img_pil)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

        self.root.after(10, self.update_video)

    def capture_as_reference(self):
        if not self.video_running:
            messagebox.showinfo("Info", "La cámara no está activa")
            return

        success, img = self.video_capture.read()
        if not success:
            messagebox.showerror("Error", "No se pudo capturar imagen")
            return

        faces = self.face_app.get(img)
        if len(faces) == 0:
            messagebox.showwarning("Advertencia", "No se detectó ningún rostro")
            return

        face = faces[0]
        geom = extract_geometric_features(face)

        dialog = tk.Toplevel(self.root)
        dialog.title("Registrar rostro")
        dialog.geometry("340x200")
        dialog.configure(bg=COLOR_CARD)

        tk.Label(dialog, text="Nombre de la persona:", bg=COLOR_CARD, font=("Segoe UI", 10)).pack(pady=(18, 6))
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack()

        tk.Label(
            dialog,
            text="Si el nombre ya existe, se agrega\ncomo una captura adicional.\n"
                 "Recuerda reentrenar el modelo después.",
            bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9), justify="center",
        ).pack(pady=(10, 10))

        def save():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Advertencia", "Ingrese un nombre")
                return
            is_new = name not in self.engine.known_faces
            self.engine.known_faces.setdefault(name, []).append({"embedding": face.embedding, "geom": geom})
            save_known_faces(self.engine.known_faces)
            self.refresh_person_lists()
            if hasattr(self, "reference_person"):
                self.reference_person.set(name)
            total = len(self.engine.known_faces[name])
            if is_new:
                self.update_result_text(f"Rostro nuevo registrado: {name} (1 captura)\n")
            else:
                self.update_result_text(f"Captura agregada a {name} (total: {total})\n")
            dialog.destroy()

        ttk.Button(dialog, text="Guardar", style="Accent.TButton", command=save).pack(pady=10)

    def register_new_face(self):
        self.capture_as_reference()

    # ==================================================================
    # Gestión de personas / dataset
    # ==================================================================

    def delete_selected_person(self):
        selection = self.people_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Selecciona una persona de la lista")
            return
        name = self.people_tree.item(selection[0])["values"][0]
        confirm = messagebox.askyesno(
            "Confirmar",
            f"¿Eliminar todas las capturas de '{name}'?\n\n"
            "Esto no borra el clasificador ya entrenado; "
            "reentrena después de eliminar para que el modelo lo olvide.",
        )
        if not confirm:
            return
        self.engine.known_faces.pop(name, None)
        save_known_faces(self.engine.known_faces)
        self.refresh_person_lists()
        self.update_result_text(f"Persona eliminada: {name}\n")

    def import_dataset_folder(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta del dataset (subcarpetas = personas)")
        if not folder:
            return
        resumen, total = import_dataset_from_path(self.engine.known_faces, folder, self.face_app)
        if total == 0:
            messagebox.showwarning(
                "Sin resultados",
                "No se detectaron rostros válidos en la carpeta seleccionada.\n"
                "Verifica que tenga subcarpetas por persona con imágenes.",
            )
            return
        self.refresh_person_lists()
        self.update_result_text(
            "\nDataset importado:\n" + "\n".join(resumen) +
            "\n(Recuerda entrenar el modelo para que use estos datos)\n"
        )
        messagebox.showinfo("Importación completa", f"Se agregaron {total} capturas en total.")

    def import_from_video(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar video", filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not filepath:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Nombre de la persona en el video")
        dialog.geometry("340x180")
        dialog.configure(bg=COLOR_CARD)

        tk.Label(dialog, text="Nombre de la persona:", bg=COLOR_CARD, font=("Segoe UI", 10)).pack(pady=(18, 6))
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack()
        tk.Label(dialog, text="El video debe mostrar solo a esta persona.",
                 bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(pady=(10, 10))

        def process():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Advertencia", "Ingrese un nombre")
                return
            dialog.destroy()
            added = extract_embeddings_from_video(self.engine.known_faces, filepath, name, self.face_app)
            if added == 0:
                messagebox.showwarning("Sin resultados", "No se detectó ningún rostro en el video")
                return
            self.refresh_person_lists()
            self.update_result_text(
                f"\nVideo procesado para '{name}': {added} capturas agregadas\n"
                f"(Recuerda entrenar el modelo para que use estos datos)\n"
            )
            messagebox.showinfo("Importación completa", f"Se agregaron {added} capturas para '{name}'.")

        ttk.Button(dialog, text="Procesar video", style="Accent.TButton", command=process).pack(pady=10)

    # ==================================================================
    # Entrenamiento del clasificador
    # ==================================================================

    def train_classifier(self):
        try:
            result = train_classifier_from_samples(self.engine.known_faces, MIN_SAMPLES_PER_PERSON)
        except ValueError as e:
            messagebox.showwarning("Advertencia", str(e))
            return

        if result["personas_insuficientes"]:
            detalle = "\n".join(f"   {n}: {c} captura(s)" for n, c in result["personas_insuficientes"].items())
            messagebox.showwarning(
                "Muy pocas muestras por persona",
                f"Se recomienda un mínimo de {MIN_SAMPLES_PER_PERSON} capturas por persona\n"
                f"(idealmente con distintos ángulos, iluminación y expresiones)\n"
                f"para que el modelo aprenda rasgos que generalicen:\n\n{detalle}\n\n"
                f"Puedes seguir entrenando de todas formas, pero es normal que\n"
                f"confunda personas o falle con caras nuevas hasta que agregues más.",
            )
            self.update_result_text(
                "Advertencia: entrenando con pocas muestras para algunas personas:\n" + detalle + "\n"
            )

        self.engine.classifier = result["model"]
        self.engine.class_labels = result["class_labels"]
        self.engine.feature_mean = result["feature_mean"]
        self.engine.feature_std = result["feature_std"]
        self.refresh_classifier_status()

        self.update_result_text(
            f"\nModelo entrenado con {result['n_samples']} muestras de {len(result['class_labels'])} personas.\n"
            f"Pérdida final (cross-entropy): {result['final_loss']:.4f}\n"
            f"Precisión sobre datos de entrenamiento: {result['train_acc']:.1f}%\n"
            f"(esto NO mide qué tan bien generaliza a caras/fotos nuevas;\n"
            f" usa la sección 'Evaluación' en video en vivo para eso)\n"
        )
        messagebox.showinfo(
            "Entrenamiento completo",
            f"Clasificador entrenado con {result['n_samples']} muestras de {len(result['class_labels'])} personas.\n"
            f"Pérdida final: {result['final_loss']:.4f}\n"
            f"Precisión en entrenamiento: {result['train_acc']:.1f}%",
        )

    # ==================================================================
    # Evaluación de métricas
    # ==================================================================

    def toggle_evaluation(self):
        if not self.eval_active:
            person = self.eval_person.get().strip()
            if not person:
                messagebox.showwarning("Advertencia", "Seleccione la persona real frente a la cámara")
                return
            if person not in self.engine.known_faces:
                messagebox.showwarning("Advertencia", "Esa persona no está registrada")
                return

            self.eval_ground_truth = person
            self.eval_active = True
            self.eval_counts = {"correct": 0, "total": 0}
            self.eval_frame_counter = 0
            self.eval_button.config(text="Detener evaluación")
            self.eval_status_label.config(text=f"Evaluando: {person} (0/0)")
            self.update_result_text(f"\nEvaluación iniciada para: {person}\n")

            if not self.video_running:
                self.update_result_text("Encendiendo cámara para la evaluación...\n")
                self.start_video()
        else:
            self.eval_active = False
            self.eval_button.config(text="Iniciar evaluación")
            total = self.eval_counts["total"]
            correct = self.eval_counts["correct"]
            acc = (correct / total * 100) if total else 0
            self.eval_status_label.config(text="Evaluación: detenida", foreground=COLOR_MUTED)
            self.update_result_text(
                f"Evaluación detenida. Resultado: {correct}/{total} aciertos ({acc:.1f}%)\n"
                f"   (registro guardado en {METRICS_LOG_PATH}, fuente='vivo')\n"
            )

            # La cámara se mantenía encendida solo para la evaluación;
            # si el usuario no está viendo "Reconocimiento", se apaga
            # para no seguir consumiendo CPU en segundo plano.
            if self.active_section != "reconocimiento":
                self.stop_video()

    def run_dataset_evaluation(self):
        if len(self.engine.known_faces) == 0:
            messagebox.showwarning("Advertencia", "No hay personas registradas todavía para comparar contra el dataset.")
            return

        folder = filedialog.askdirectory(
            title="Seleccionar carpeta del dataset a evaluar (subcarpetas = personas)"
        )
        if not folder:
            return

        self.dataset_eval_status_label.config(text="Evaluando... esto puede tardar según el tamaño del dataset.")
        self.root.update_idletasks()

        summary = evaluate_dataset(self.engine, folder, self.face_app, only_registered=True)

        if summary["total"] == 0:
            self.dataset_eval_status_label.config(text="No se evaluó ninguna imagen (revisa la carpeta).")
            messagebox.showwarning(
                "Sin resultados",
                "No se evaluó ninguna imagen. Verifica que la carpeta tenga subcarpetas "
                "con el mismo nombre que las personas ya registradas, y que contengan fotos "
                "donde se detecte un rostro.",
            )
            return

        acc = summary["accuracy"]
        self.dataset_eval_status_label.config(
            text=f"Última evaluación: {summary['correct']}/{summary['total']} "
                 f"aciertos ({acc:.1f}%)"
        )

        lines = ["\n" + "=" * 40, "EVALUACIÓN SOBRE DATASET:", f"   Carpeta: {folder}",
                 f"   Total de imágenes evaluadas: {summary['total']}",
                 f"   Precisión global: {acc:.1f}% ({summary['correct']}/{summary['total']})"]

        if summary["sin_rostro"]:
            lines.append(f"   Imágenes sin rostro detectado (omitidas): {summary['sin_rostro']}")
        if summary["carpetas_omitidas"]:
            lines.append(
                "   Carpetas omitidas (sin registrar como persona): "
                + ", ".join(summary["carpetas_omitidas"])
            )

        lines.append("   Por persona:")
        for name, stats in sorted(summary["per_person"].items()):
            lines.append(f"      {name}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")

        # -- Métricas estilo deep learning: no solo si acertó, sino
        # cómo se comporta la red (pérdida, calibración, velocidad) --
        lines.append("")
        lines.append("   Comportamiento de la red (más allá de acierto/fallo):")
        if summary["avg_loss"] is not None:
            lines.append(f"      Pérdida promedio (cross-entropy sobre clase real): {summary['avg_loss']:.4f}")
        else:
            lines.append("      Pérdida: no disponible (sin clasificador entrenado)")

        cc, ci = summary["confidence_correct"], summary["confidence_incorrect"]
        if cc["n"]:
            lines.append(f"      Confianza promedio en ACIERTOS: {cc['mean']*100:.1f}% (± {cc['std']*100:.1f}, n={cc['n']})")
        if ci["n"]:
            lines.append(f"      Confianza promedio en FALLOS:   {ci['mean']*100:.1f}% (± {ci['std']*100:.1f}, n={ci['n']})")
        if cc["n"] and ci["n"]:
            if cc["mean"] > ci["mean"]:
                lines.append("      -> El modelo tiende a estar más seguro cuando acierta (buena señal de calibración).")
            else:
                lines.append("      -> Aviso: el modelo no está más seguro en aciertos que en fallos (mala calibración).")

        if summary["avg_inference_ms"] is not None:
            lines.append(
                f"      Tiempo de inferencia: {summary['avg_inference_ms']:.1f} ms promedio "
                f"(máx {summary['max_inference_ms']:.1f} ms) por imagen"
            )

        curve = summary["error_curve"]
        if len(curve) >= 2:
            mitad = len(curve) // 2
            error_primera_mitad = curve[mitad - 1][1]
            error_total = curve[-1][1]
            lines.append(
                f"      Error acumulado: {error_primera_mitad:.1f}% a la mitad del recorrido, "
                f"{error_total:.1f}% al final (detalle punto a punto en el CSV)"
            )

        # -- Análisis de robustez ante oclusión (mascarilla) --
        upper = summary["upper_face_analysis"]
        if upper["n_personas_con_perfil"] > 0:
            lines.append("")
            lines.append("   Robustez ante oclusión (solo cejas/ojos, útil para mascarilla):")
            us_c = upper["similitud_superior_aciertos"]
            us_i = upper["similitud_superior_fallos"]
            if us_c["n"]:
                lines.append(f"      Similitud superior en ACIERTOS: {us_c['mean']*100:.1f}% (n={us_c['n']})")
            if us_i["n"]:
                lines.append(f"      Similitud superior en FALLOS:   {us_i['mean']*100:.1f}% (n={us_i['n']})")
            if us_i["n"] and us_i["mean"] is not None and us_i["mean"] > 0.7:
                lines.append(
                    "      -> Los fallos igual muestran alta similitud en cejas/ojos: la mitad "
                    "superior del rostro seguía siendo reconocible pese al error principal."
                )

        if summary["report_path"]:
            lines.append("")
            lines.append(f"   Reporte detallado por imagen (para graficar en Excel): {summary['report_path']}")

        lines.append(f"   (registro resumido también en {METRICS_LOG_PATH}, fuente='dataset')")
        lines.append("=" * 40)

        self.update_result_text("\n".join(lines) + "\n")

        messagebox.showinfo(
            "Evaluación de dataset completa",
            f"Precisión global: {acc:.1f}% ({summary['correct']}/{summary['total']})\n\n"
            f"Detalle completo (pérdida, calibración, tiempos, robustez ante\n"
            f"oclusión) en el panel de actividad y en:\n{summary['report_path']}",
        )

    # ==================================================================
    # Cierre de la aplicación
    # ==================================================================

    def quit_app(self):
        self.stop_video()
        self.root.quit()
        self.root.destroy()
