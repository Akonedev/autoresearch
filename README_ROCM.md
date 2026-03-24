# AMD ROCm Setup for Autoresearch

Ce document explique comment configurer et utiliser autoresearch avec des GPU AMD ROCm.

## Configuration système requise

- GPU AMD RDNA2/RDNA3 ou AMD Instinct (MI200/MI300 series)
- ROCm 6.0+ installé (vérifié avec `rocm-smi --showproductname`)
- Python 3.10+
- PyTorch avec support ROCm

## Installation

### 1. Installer les dépendances

```bash
# Supprimer l'ancien environnement et lock file
rm -rf .venv uv.lock

# Installer les dépendances (sans torch)
uv sync --no-dev

# Installer PyTorch ROCm séparément
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.0 --index-strategy unsafe-best-match
```

### 2. Vérifier l'installation

```bash
uv run python -c "import torch; print('ROCm:', torch.version.hip); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Différences avec NVIDIA CUDA

### Support ROCm implémenté

✅ **Détection automatique** - Le code détecte automatiquement ROCm vs CUDA
✅ **PyTorch SDPA** - Utilise `scaled_dot_product_attention` natif de PyTorch (dispatche vers AOTriton)
✅ **Fallback CPU** - Fonctionne même sans GPU
✅ **MFU calculation** - Valeurs de peak FLOPS pour les GPU AMD Radeon et Instinct

### Limitations actuelles

⚠️ **Flash Attention 3** - Non disponible sur ROCm, utilise SDPA native
⚠️ **torch.compile** - Désactivé par défaut (peut fonctionner avec PyTorch 2.9+ ROCm)
⚠️ **Window Attention** - Le pattern "SSSL" dégénère en attention causale complète sur ROCm

### Paramètres recommandés pour AMD GPUs

Les paramètres par défaut dans `train.py` sont optimisés pour RX 7900 XTX (24GB VRAM) :

```python
DEPTH = 6               # 6 couches (vs 8 pour NVIDIA)
DEVICE_BATCH_SIZE = 32  # Batch size réduit (vs 128)
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens (vs 524K)
WINDOW_PATTERN = "L"    # Pattern simple (vs "SSSL")
```

**Pour adapter à votre GPU :**

| GPU | VRAM | DEPTH | DEVICE_BATCH_SIZE |
|-----|------|-------|-------------------|
| RX 7900 XTX | 24GB | 6 | 32 |
| RX 7900 XT | 20GB | 6 | 24 |
| RX 7800 XT | 16GB | 4 | 16 |
| RX 6900 XT | 16GB | 4 | 16 |
| MI250X | 128GB | 8 | 64 |
| MI300X | 192GB | 12 | 128 |

## Utilisation

### 1. Préparer les données

```bash
uv run python prepare.py --num-shards 10  # Télécharger 10 shards
```

### 2. Lancer l'entraînement

```bash
uv run python train.py
```

L'entraînement dure 5 minutes et produit :
- `val_bpb` : Bits per byte (métrique d'évaluation)
- `mfu_percent` : Model FLOPs Utilization
- `peak_vram_mb` : Mémoire GPU maximale utilisée

### 3. Interpréter les résultats

Exemple de sortie sur RX 7900 XTX :
```
Using GPU 0: Radeon RX 7900 XTX
Detected GPU: Radeon RX 7900 XTX -> peak BF16 FLOPS: 6.1e+14
...
step 00320 (100.0%) | loss: 3.456789 | lrm: 0.50 | dt: 850ms | tok/sec: 154,260 | mfu: 3.5%
---
val_bpb:          3.234567
training_seconds: 300.0
mfu_percent:      3.50
```

**MFU typique sur AMD GPUs :** 3-5% (SDPA sans Flash Attention)

## Dépannage

### Out of Memory (OOM)

Réduisez `DEVICE_BATCH_SIZE` et/ou `DEPTH` dans `train.py` :

```python
DEVICE_BATCH_SIZE = 16  # Réduire si OOM
DEPTH = 4               # Réduire le nombre de couches
```

### Performance faible

Vérifiez que ROCm est correctement installé :

```bash
rocm-smi --showproductname
python3 -c "import torch; print(torch.version.hip)"
```

### torch.compile ne fonctionne pas

C'est normal. Sur ROCm, `torch.compile` est désactivé par défaut car le support est expérimental.

## Références

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/docs/stable/notes/rocm.html)
- [AMD Instinct GPUs](https://www.amd.com/en/products/accelerators.html)
