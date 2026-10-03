"""
Paleta de colores y configuración de estilos ttk, centralizadas para
que toda la interfaz tenga una apariencia consistente y profesional
en vez de los colores planos/por defecto de Tkinter.
"""

import tkinter as tk
from tkinter import ttk

# ------------------------------------------------------------------
# Paleta
# ------------------------------------------------------------------
COLOR_BG = "#f4f6fa"           # fondo general de la ventana
COLOR_SIDEBAR = "#1c2537"      # barra de navegación lateral (oscura)
COLOR_SIDEBAR_TEXT = "#e6eaf2"
COLOR_ACCENT = "#3f7cf0"       # azul de acción / selección
COLOR_ACCENT_DARK = "#2f5fce"
COLOR_SUCCESS = "#1f9d55"
COLOR_DANGER = "#e0464b"
COLOR_MUTED = "#8892a4"
COLOR_CARD = "#ffffff"         # tarjetas / paneles de contenido
COLOR_BORDER = "#dde3ee"
COLOR_TEXT = "#1c2434"

FONT_FAMILY = "Segoe UI"


def apply_theme(root):
    """Configura la paleta y los estilos ttk que usa toda la app.
    Debe llamarse una sola vez, apenas se crea la ventana raíz."""
    root.configure(bg=COLOR_BG)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass  # si 'clam' no está disponible, se usa el tema por defecto

    style.configure(".", font=(FONT_FAMILY, 10), background=COLOR_BG, foreground=COLOR_TEXT)

    # Contenedores
    style.configure("TFrame", background=COLOR_BG)
    style.configure("Card.TFrame", background=COLOR_CARD)
    style.configure("Sidebar.TFrame", background=COLOR_SIDEBAR)

    # Etiquetas
    style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT)
    style.configure("Header.TLabel", font=(FONT_FAMILY, 17, "bold"),
                    background=COLOR_BG, foreground=COLOR_TEXT)
    style.configure("MutedOnBg.TLabel", foreground=COLOR_MUTED, background=COLOR_BG)
    style.configure("SectionTitle.TLabel", font=(FONT_FAMILY, 13, "bold"),
                    background=COLOR_CARD, foreground=COLOR_TEXT)
    style.configure("Muted.TLabel", foreground=COLOR_MUTED, background=COLOR_CARD)

    # Botones
    style.configure("TButton", font=(FONT_FAMILY, 10), padding=8)
    style.configure("Accent.TButton", font=(FONT_FAMILY, 10, "bold"), padding=(14, 9),
                     background=COLOR_ACCENT, foreground="white", borderwidth=0)
    style.map(
        "Accent.TButton",
        background=[("active", COLOR_ACCENT_DARK), ("disabled", COLOR_MUTED)],
        foreground=[("disabled", "#eef1f7")],
    )
    style.configure("Danger.TButton", font=(FONT_FAMILY, 10, "bold"), padding=(14, 9),
                     background=COLOR_DANGER, foreground="white", borderwidth=0)
    style.map("Danger.TButton", background=[("active", "#c23a3f")])

    # Notebook / pestañas (no se usan en el layout final, pero quedan
    # definidas por si se reutilizan en el futuro)
    style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
    style.configure("TNotebook.Tab", font=(FONT_FAMILY, 10, "bold"), padding=(16, 10))

    # Marcos con título (paneles tipo "Advertencia", secciones, etc.)
    style.configure("TLabelframe", background=COLOR_CARD, bordercolor=COLOR_BORDER,
                     relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=COLOR_CARD, foreground=COLOR_TEXT,
                     font=(FONT_FAMILY, 10, "bold"))

    # Controles de entrada
    style.configure("TCombobox", padding=6)
    style.configure("TEntry", padding=6)
    style.configure("Horizontal.TScale", background=COLOR_CARD)

    # Tabla (lista de personas registradas)
    style.configure("Treeview", font=(FONT_FAMILY, 10), rowheight=26,
                     background=COLOR_CARD, fieldbackground=COLOR_CARD, foreground=COLOR_TEXT)
    style.configure("Treeview.Heading", font=(FONT_FAMILY, 10, "bold"))
    style.map("Treeview", background=[("selected", COLOR_ACCENT)], foreground=[("selected", "white")])

    return style
