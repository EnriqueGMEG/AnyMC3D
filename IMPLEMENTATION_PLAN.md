# Plan de implementación: AnyMC3D para metástasis en PDAC

## Alcance

Este documento audita la reproducción pública
`EnriqueGMEG/AnyMC3D` (rama `main`) y define la adaptación para
clasificación binaria de metástasis a nivel de paciente a partir de CT
portal-venoso y una máscara fina del páncreas.

La referencia metodológica es la versión 3 de:

> Liu et al., “Revisiting 2D Foundation Models for Scalable 3D Medical
> Image Classification”, arXiv:2512.12887v3.

No existe código oficial enlazado desde el artículo. Por tanto, todo detalle
del repositorio actual que no esté descrito en el artículo se considera una
decisión propia de la reproducción y no una implementación oficial.

## Fase 1: auditoría del repositorio actual

### Arquitectura y flujo actuales

El flujo DINO actual está implementado principalmente en
`model_arch/anymc3d.py`:

1. Carga DINOv2 desde `torch.hub` (`dinov2_vitb14` o `dinov2_vitl14`).
2. Congela inicialmente el backbone.
3. PEFT añade LoRA a módulos llamados `qkv`, `proj` y
   `patch_embed.proj`.
4. Cada volumen, ya redimensionado a una shape fija por el Dataset, se
   separa en slices.
5. Cada slice se replica a tres canales o, opcionalmente, se transforma en
   una entrada 2.5D.
6. Cada slice se vuelve a redimensionar a un cuadrado `input_size ×
   input_size`.
7. Se extraen `x_norm_clstoken` y `x_norm_patchtokens` mediante
   `forward_features`.
8. Opcionalmente se aplican dos bloques Transformer entrenables nuevos que
   no aparecen en AnyMC3D.
9. Un task query agrega los embeddings de slices mediante atención.
10. Una cabeza `Dropout + Linear` produce logits de clase.

La configuración PDCAD activa CLS-only, slice attention y desactiva los
bloques Transformer adicionales, el input 2.5D y la concatenación de patch
tokens.

### Coincidencias con el artículo

- Backbone 2D congelado y adaptación LoRA.
- Rank LoRA 8 y alpha 16.
- LoRA dirigida, para la implementación DINOv2 de `torch.hub`, a patch
  embedding, QKV fusionada y proyección de salida.
- Replicación de slices monocanal a RGB.
- Normalización ImageNet explícita.
- Extracción de un CLS embedding por slice.
- Query-based attention pooling con escala `sqrt(D)`.
- Task query con inicialización truncated normal, `std=0.02`.
- Predicción y focal loss calculadas después de la fusión, a nivel de
  volumen/paciente; no existe loss por slice.
- Grupos de optimización separados para LoRA y cabeza/query con los learning
  rates y weight decays del apéndice A.
- Augmentation 3D basada en el apéndice A3.

### Diferencias y decisiones propias de la reproducción

| Área | Repositorio actual | Artículo / pipeline solicitado |
|---|---|---|
| Backbone | DINOv2 por `torch.hub` | DINOv3 oficial de Transformers |
| CLS | Diccionario propio de DINOv2 (`x_norm_clstoken`) | Posición 0 de `last_hidden_state`, verificada para DINOv3 |
| LoRA | Nombres DINOv2 hardcodeados; PEFT | Verificar nombres reales DINOv3; fallar si falta un objetivo |
| Adaptador extra | Hasta dos bloques Transformer entrenables | No forman parte del método descrito; no se usarán |
| Patch features | Concatenación/pooling opcional | CLS de última capa únicamente |
| Cabeza binaria | Dos logits + softmax | Un logit + sigmoid solo para métricas/inferencia |
| Crop | Bounding box de CT no-cero | Bounding box rectangular de máscara pancreática |
| Margen | Vóxeles | Milímetros convertidos por eje con `ceil` |
| Geometría | Affine descartada; `scipy.zoom` | Affine y campo físico preservados |
| Shape | Resize trilineal fijo H×W×S | Spacing común + padding H×W, S variable |
| Slices | Se fuerza `S=70` en PDCAD | Todas las slices reales + `slice_mask` |
| Input DINO | Resize cuadrado oculto dentro del forward | Canvas rectangular divisible por 16, sin resize |
| Batch variable | No soportado | Collate con padding exclusivo en el eje S |
| Atención | Sin máscara de slices | Softmax enmascarado; padding con peso cero |
| Padding slices | Las slices artificiales serían codificadas | Solo se codifican slices válidas |
| Normalización CT | Percentiles genéricos | Clip HU `[-150, 250]` y rescale `[0, 1]` |
| Augmentation | Flips, zoom y cambios de escala por defecto | Perfil `size_preserving` por defecto |
| Métricas | AUROC, accuracy, F1 y balanced accuracy | Métricas clínicas completas, PR-AUC y Brier |
| Checkpoint | `val/AUROC` hardcodeado | Métrica y dirección configurables |
| Exportación | No guarda predicciones/atención durante CV | CSV por fold, atención, geometría y OOF |
| Inferencia | Duplica funciones; permite optimizar umbral en el conjunto inferido | Reutiliza pipeline compartido y umbral solo de validation |
| Errores de datos | Omite casos sin label/fichero con warning | Validación estricta; nunca descartar casos silenciosamente |
| Rutas | Varias rutas absolutas en YAML y ejemplos Python | Rutas configurables/relativas |

### Riesgos detectados

- `ClassificationDataset._load_volume` deforma anatomía y fuerza el número
  de slices mediante interpolación trilineal.
- `ModalityEncoder.forward` aplica además resize bilineal cuadrado en plano.
- `crop_to_nonzero` puede confundir aire/fondo y no representa el páncreas.
- El preprocesamiento no conserva un affine actualizado ni coordenadas
  físicas de slices.
- CT y máscara no se validan ni se alinean porque la máscara no participa.
- Los casos sin label o fichero se eliminan silenciosamente después de un
  warning.
- No existe comprobación explícita de leakage por `patient_id`.
- La inferencia puede optimizar Youden sobre el propio conjunto etiquetado
  inferido; esto sería leakage si se usa con test.
- No existen tests automatizados.
- La configuración ViT-L actual no tiene el `_target_` requerido por Hydra y
  no es equivalente a la configuración ViT-B.

## Decisiones conservadoras para la nueva implementación

1. Mantener los módulos históricos para no romper experimentos DINOv2/V-JEPA,
   pero añadir un camino separado y explícito para metástasis pancreática.
2. Usar NIfTI canónico RAS y tratar los ejes de arrays como `(X, Y, Z)`.
   El tensor del modelo será `(S, 1, H, W) = (Z, 1, Y, X)`.
3. Validar CT/máscara antes y después de la canonicalización. Por defecto,
   cualquier desalineamiento es fatal.
4. Si se autoriza `resample_mask_to_ct`, usar exclusivamente nearest-neighbor.
5. Reorientar y, si procede, reamostrar CT y máscara antes de calcular el
   bounding box. La máscara se descarta tras localizar el crop.
6. Calcular el canvas únicamente con geometría y máscaras de todo el
   manifiesto; no leer labels para esa decisión.
7. Persistir spacing auto-resuelto y canvas en `geometry_config.json`; nunca
   recalcularlos durante training o inference.
8. Usar padding constante de valor 0 después de la ventana HU.
9. Usar un único logit por paciente y `BCEWithLogitsLoss` o focal binaria
   basada en logits.
10. No introducir embeddings posicionales entre slices: las posiciones
    físicas se conservan solo como metadatos exportables.

## Fases de implementación

### Fase 2: geometría y preprocesamiento

- Crear utilidades compartidas en `preprocessing/`:
  - carga y canonicalización;
  - validación de alineamiento;
  - resampling físico con actualización de affine;
  - bounding box de máscara y margen en mm;
  - ventana HU;
  - padding determinista;
  - auditoría de spacing/canvas y memoria.
- Crear `analyze_dataset_geometry.py`.
- Crear `preprocess_dataset.py`.
- Generar y reutilizar `geometry_config.json`.
- Guardar log JSONL por caso y `patient_geometry.csv`.

### Fase 3: DINOv3 y LoRA

- Cargar `Dinov3Model`/`AutoModel` desde Transformers sin
  `AutoImageProcessor`.
- Verificar Transformers >= 4.56 y errores de acceso/licencia.
- Inspeccionar y registrar nombres exactos de módulos.
- Implementar wrappers LoRA para `Linear` y `Conv2d`, con base congelada y
  actualización inicialmente nula.
- Adaptar patch projection, Q, K, V y output projection.
- Extraer CLS como `last_hidden_state[:, 0, :]`.
- Codificar solo slices reales, en chunks configurables.

### Fase 4: dataset variable

- Leer el manifiesto único:
  `patient_id,ct_path,pancreas_mask_path,label,fold`.
- Validar de forma estricta paths, labels, duplicados, folds, geometría y
  leakage.
- Cargar artefactos preprocesados sin resize.
- Implementar collate a `B × S_max × 1 × H × W`, `slice_mask` y metadatos.
- Añadir perfiles 3D coherentes `size_preserving` y `paper_like`.

### Fase 5: entrenamiento, CV e inferencia

- Añadir LightningModule binario con métricas a nivel de paciente.
- Checkpoint configurable por `val_pr_auc`, `val_auroc` o `val_loss`.
- Exportar predicciones, atención y geometría por fold.
- Crear `train_cv.py` y agregación OOF.
- Crear `inference.py` que reutilice exactamente el preprocesamiento común y
  el `geometry_config.json` persistido.

### Fase 6: validación

- Añadir tests unitarios y smoke tests sintéticos para los 32 criterios.
- Ejecutar auditoría sobre las NIfTI de prueba suministradas.
- Ejecutar `pytest -q`.
- Ejecutar un forward real de DINOv3 cuando los pesos estén accesibles.
- Informar parámetros totales/entrenables, módulos LoRA y shapes observadas.

## Limitaciones conocidas antes de implementar

- Las carpetas de prueba aportan CT y máscaras, pero no aportan etiquetas de
  metástasis ni folds. Son suficientes para la auditoría geométrica y smoke
  tests de preprocesamiento; el entrenamiento real requerirá el manifiesto
  completo.
- Los checkpoints DINOv3 de Facebook en Hugging Face pueden requerir aceptar
  licencia y autenticación. Nunca se sustituirá silenciosamente el backbone.
- El artículo reporta shape fija para T5 y no especifica el margen del crop.
  Spacing común, margen configurable en mm, canvas con padding y S variable
  son decisiones intencionadas de este proyecto.
