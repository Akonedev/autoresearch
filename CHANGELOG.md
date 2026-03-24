# Changelog - AMD ROCm Support

## Version 1.0.0 - Support Multi-GPU (AMD ROCm + NVIDIA CUDA + CPU)

### 🎉 Nouveautés principales

#### Support AMD ROCm complet
- ✅ Détection automatique AMD ROCm vs NVIDIA CUDA
- ✅ Support des GPU AMD Radeon (RX 6000/7000 series)
- ✅ Support des GPU AMD Instinct (MI200/MI300 series)
- ✅ Fallback automatique vers CPU si aucun GPU disponible
- ✅ Calcul MFU adapté pour les GPU AMD

#### Optimisations mémoire
- ✅ Gradient checkpointing (réduit la mémoire de ~40-50%)
- ✅ Allocation dynamique de mémoire (expandable_segments)
- ✅ Pinned memory uniquement si GPU disponible
- ✅ Gestion intelligente VRAM pour éviter OOM

#### Performance
- ✅ PyTorch SDPA (Scaled Dot Product Attention) pour ROCm
- ✅ Support expérimental torch.compile sur ROCm 7.0+
- ✅ Peak FLOPS configurables par modèle de GPU
- ✅ Batch size et profondeur ajustables

---

### 📝 Changements détaillés

#### Fichier: `train.py`

**Ajouts:**
- Import `checkpoint` de `torch.utils.checkpoint`
- Détection `IS_ROCM` et `IS_CUDA`
- Message d'information ROCm au démarrage
- Affichage du nom du GPU détecté
- Peak FLOPS pour AMD Radeon et Instinct
- Gradient checkpointing (configurable via `USE_GRADIENT_CHECKPOINTING`)
- Support torch.compile expérimental pour ROCm (configurable via `USE_TORCH_COMPILE`)
- Synchronisations GPU conditionnelles (`torch.cuda.is_available()`)
- Gestion mémoire VRAM conditionnelle

**Modifications:**
- `WINDOW_PATTERN` par défaut: `"L"` (plus simple pour ROCm)
- `TOTAL_BATCH_SIZE`: `2**17` (réduit de `2**19`)
- `DEVICE_BATCH_SIZE`: `32` (réduit de `128`)
- `DEPTH`: `6` (réduit de `8`)

**Nouvelles configurations:**
```python
USE_GRADIENT_CHECKPOINTING = False  # Économise mémoire, plus lent
USE_TORCH_COMPILE = False  # Expérimental sur ROCm 7.0+
```

#### Fichier: `prepare.py`

**Modifications:**
- `get_token_bytes()`: Device dynamique (GPU ou CPU)
- `make_dataloader()`: 
  - Pinned memory uniquement si GPU disponible
  - Buffer GPU alloué conditionnellement
  - Fallback CPU automatique
- `evaluate_bpb()`: Device dynamique pour token_bytes

#### Fichier: `pyproject.toml`

**Changements:**
- Suppression de la dépendance explicite `torch==2.9.1` (CUDA only)
- PyTorch installé séparément selon la plateforme
- Support Python 3.10+
- Index ROCm: `https://download.pytorch.org/whl/rocm7.0`

#### Fichier: `README.md`

**Ajouts:**
- Section "Platform Support" mise à jour
- Lien vers `README_ROCM.md`
- Instructions pour AMD GPUs

#### Nouveau fichier: `README_ROCM.md`

Guide complet pour AMD ROCm:
- Installation étape par étape
- Configuration recommandée par GPU
- Limitations et workarounds
- Dépannage
- Références

#### Nouveau fichier: `OPTIMIZATION_GUIDE.md`

Guide d'optimisation avancée:
- Flash Attention pour ROCm
- Kernels Triton custom
- Optimisations mémoire (gradient checkpointing)
- torch.compile sur ROCm
- Benchmark et profiling
- Roadmap des améliorations

---

### 🔧 Installation

#### NVIDIA CUDA (inchangé)
```bash
uv sync
uv run python prepare.py
uv run python train.py
```

#### AMD ROCm (nouveau)
```bash
# Supprimer ancien environnement
rm -rf .venv uv.lock

# Installer dépendances
uv sync --no-dev

# Installer PyTorch ROCm
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.0 \
  --index-strategy unsafe-best-match

# Utiliser
uv run python prepare.py
uv run python train.py
```

#### CPU (fallback)
```bash
uv sync
uv run python prepare.py
uv run python train.py  # Automatiquement en mode CPU
```

---

### 📊 Performance

#### RX 7900 XTX (24GB VRAM) - Configuration par défaut
- **Modèle:** 26M paramètres, 6 couches
- **Batch size:** 32 (device), 131K (total)
- **Performance:** ~150K tokens/sec
- **MFU:** ~3.5% (SDPA sans Flash Attention)
- **VRAM utilisée:** ~21GB

#### Avec optimisations (à venir)
- **Gradient checkpointing activé:** Batch size 64 possible
- **Flash Attention 2 (future):** MFU ~8-12%
- **Triton kernels (future):** +20-40% MLP

---

### ⚠️ Limitations connues

#### ROCm
1. **Flash Attention 3** - Non disponible, utilise SDPA native
2. **torch.compile** - Expérimental, désactivé par défaut
3. **Window Attention** - Pattern "SSSL" dégénère en attention complète
4. **MFU** - Plus bas que NVIDIA (~3-5% vs ~10-15%)

#### Workarounds
- Utiliser `WINDOW_PATTERN = "L"` (défaut actuel)
- Activer `USE_GRADIENT_CHECKPOINTING = True` pour plus de mémoire
- Réduire `DEVICE_BATCH_SIZE` si OOM

---

### 🎯 Configuration recommandée par GPU

| GPU AMD | VRAM | DEPTH | DEVICE_BATCH_SIZE | TOTAL_BATCH_SIZE |
|---------|------|-------|-------------------|------------------|
| RX 7900 XTX | 24GB | 6 | 32 | 2^17 |
| RX 7900 XT | 20GB | 6 | 24 | 2^17 |
| RX 7800 XT | 16GB | 4 | 16 | 2^16 |
| RX 6900 XT | 16GB | 4 | 16 | 2^16 |
| RX 6800 XT | 16GB | 4 | 16 | 2^16 |
| MI250X | 128GB | 8 | 64 | 2^18 |
| MI300X | 192GB | 12 | 128 | 2^19 |

---

### 🐛 Bugs corrigés

- ❌ OOM immédiat sur AMD GPUs → ✅ Batch size réduit
- ❌ Crash sans GPU → ✅ Fallback CPU automatique
- ❌ torch.cuda.synchronize() crash → ✅ Conditionnel
- ❌ Peak FLOPS incorrect → ✅ Valeurs AMD ajoutées

---

### 📚 Documentation

- `README.md` - Guide principal
- `README_ROCM.md` - Guide spécifique AMD ROCm
- `OPTIMIZATION_GUIDE.md` - Guide d'optimisation avancée
- `CHANGELOG.md` - Ce fichier

---

### 🔮 Roadmap

#### v1.1.0 (Prochainement)
- [ ] Flash Attention 2 depuis source
- [ ] Triton kernels pour MLP
- [ ] torch.compile stable sur ROCm

#### v1.2.0 (Futur)
- [ ] Multi-GPU support
- [ ] Profiling automatique
- [ ] Auto-tuning des hyperparamètres

---

### 🙏 Remerciements

- @karpathy pour autoresearch original
- @andyluo7 pour le fork AMD initial
- Communauté ROCm pour le support PyTorch
- Contributeurs AMD et NVIDIA

---

### 📄 License

MIT License - Voir LICENSE
