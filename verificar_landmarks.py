"""
Herramienta de verificación de landmarks faciales.

Antes de confiar en las características geométricas (ojos, nariz,
boca, mandíbula) que se calculan a partir de los 106 puntos de
referencia, es importante confirmar VISUALMENTE qué índice corresponde
a qué parte de la cara, ya que puede variar levemente según la
versión de insightface instalada.

Uso:
    python verificar_landmarks.py ruta_a_una_foto.jpg

Esto genera 'landmarks_debug.jpg' con los 106 puntos numerados
dibujados sobre la imagen. Ábrelo y confirma que:
    - Puntos ~0-32   están en el contorno de la mandíbula/cara
    - Puntos ~33-50  están en las cejas
    - Puntos ~51-65  están en la nariz
    - Puntos ~66-85  están en los ojos
    - Puntos ~86-105 están en la boca / labios

Si los rangos no coinciden con lo que ves en la imagen, avísame los
rangos correctos y ajusto la función extract_geometric_features().
"""

import sys
import cv2
import insightface

def main():
    if len(sys.argv) < 2:
        print("Uso: python verificar_landmarks.py ruta_a_una_foto.jpg")
        return

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return

    app = insightface.app.FaceAnalysis()
    app.prepare(ctx_id=-1)

    faces = app.get(img)
    if len(faces) == 0:
        print("No se detectó ningún rostro en la imagen")
        return

    face = faces[0]
    lm = face.landmark_2d_106
    if lm is None:
        print("Este modelo no generó landmark_2d_106 (revisa la instalación de insightface)")
        return

    print(f"Se detectaron {len(lm)} puntos de referencia")

    for i, (x, y) in enumerate(lm):
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        # Etiquetar cada 3 puntos para no saturar la imagen
        if i % 3 == 0:
            cv2.putText(img, str(i), (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    output_path = "landmarks_debug.jpg"
    cv2.imwrite(output_path, img)
    print(f"Guardado: {output_path}")
    print("\nAbre esa imagen y confirma que los rangos de índice coincidan con:")
    print("  0-32   -> contorno de la cara/mandíbula")
    print("  33-50  -> cejas")
    print("  51-65  -> nariz")
    print("  66-85  -> ojos")
    print("  86-105 -> boca/labios")
    print("\nSi no coincide, dime qué números ves realmente en cada zona.")


if __name__ == "__main__":
    main()
