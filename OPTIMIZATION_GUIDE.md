# Guide d'optimisation pour AMD ROCm

Ce document présente les techniques pour améliorer les performances sur GPU AMD ROCm.

---

## 1. Flash Attention pour ROCm ⚡

### Problème
Flash Attention 3 n'est pas officiellement supporté sur ROCm, mais il existe des alternatives.

### Solution A: Flash Attention 2 (support ROCm partiel)

```bash
# Installer depuis source avec support ROCm
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout flash_v2  # Version avec support ROCm

# Installer avec HIP
pip install -v -U .
```

**Gain attendu:** +50-100% sur l'attention, MFU ~8-12%

### Solution B: Utiliser `triton` avec kernels custom

```python
# Installer triton-rocm
pip install triton-rocm

# Exemple de kernel d'attention optimisé
import triton
import triton.language as tl

@triton.jit
def attention_kernel(...):
    # Kernel d'attention optimisé pour ROCm
    pass
```

**Gain attendu:** +30-50%, MFU ~6-10%

### Solution C: Utiliser `kernels-community/flash-attn3` avec patch ROCm

Certains forks communautaires ajoutent le support ROCm:

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
python setup.py install  # Avec ROCm détecté
```

---

## 2. Optimisations avec Triton 🔧

### Installer Triton pour ROCm

```bash
pip install triton-rocm
```

### Exemple: Kernel MLP optimisé

```python
import triton
import triton.language as tl

@triton.jit
def mlp_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_w, stride_y,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """MLP forward pass optimisé pour ROCm."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Charger les données
    x = tl.load(X_ptr + row_idx * stride_x + col_offsets)
    w = tl.load(W_ptr + col_offsets)
    
    # Computation
    y = tl.relu(x * w)
    
    # Store
    tl.store(Y_ptr + row_idx * stride_y + col_offsets, y)
```

**Gain attendu:** +20-40% sur les couches MLP

### Resources Triton ROCm

- [Triton Documentation ROCm](https://triton-lang.org/main/getting-started/installation.html#rocm)
- [Exemples de kernels](https://github.com/ROCm/triton-kernels)

---

## 3. Optimisations Mémoire 🧠

### A. Gradient Checkpointing

Réduit la mémoire utilisée pour permettre un batch size plus élevé.

```python
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size):
        # Utiliser checkpoint pour économiser la mémoire
        return checkpoint(
            self._forward_impl, x, ve, cos_sin, window_size,
            use_reentrant=False
        )
    
    def _forward_impl(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x
```

**Gain:** -40-50% mémoire → batch size 2x plus grand → MFU +50-100%

### B. Mixed Precision (BF16/FP16)

Déjà implémenté avec `autocast_ctx`, mais on peut optimiser:

```python
# Dans train.py
autocast_ctx = torch.amp.autocast(
    device_type="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16  # BF16 pour AMD RDNA3
)

# Pour RDNA2 (RX 6000), utiliser FP16 si BF16 lent
# dtype=torch.float16
```

### C. Activation Recycling

```python
# Libérer la mémoire des activations non nécessaires
torch.cuda.empty_cache()  # Ou torch.xpu.empty_cache()
```

### D. Augmenter le batch size progressivement

```python
# Après gradient checkpointing, augmenter:
DEVICE_BATCH_SIZE = 64  # Au lieu de 32
TOTAL_BATCH_SIZE = 2**18  # Au lieu de 2**17
```

**Gain:** +50-100% throughput → MFU proportionnel

---

## 4. torch.compile sur ROCm 🔨

### Support expérimental (PyTorch 2.10+ ROCm 7.0)

```python
# Dans train.py, remplacer:
if IS_CUDA:
    model = torch.compile(model, dynamic=False)
else:
    # Essayer torch.compile sur ROCm 7.0+
    if IS_ROCM:
        try:
            model = torch.compile(model, dynamic=False, backend="inductor")
            print("ROCm: torch.compile enabled with inductor backend")
        except Exception as e:
            print(f"ROCm: torch.compile failed ({e}), disabling")
    else:
        print("CPU mode: torch.compile disabled")
```

**Gain potentiel:** +20-30% si fonctionne

### Backend alternatives

```python
# Essayer le backend 'eager' ou 'cudagraphs'
model = torch.compile(model, backend="eager")  # Plus stable
# ou
model = torch.compile(model, backend="inductor")  # Plus rapide si stable
```

---

## 5. Autres Optimisations 🚀

### A. Data Loading optimisé

```python
# Dans prepare.py, augmenter les workers
def make_dataloader(..., num_workers=4, prefetch_factor=2):
    # Utiliser DataLoader avec workers multiples
    pass
```

### B. Fusion d'opérations

```python
# Utiliser des opérations fusionnées quand possible
x = F.rms_norm(x, (x.size(-1),))  # Déjà utilisé ✓
x = F.relu(x).square()  # Peut être fusionné
```

### C. Communication overlap (multi-GPU futur)

```python
# Pour le futur support multi-GPU AMD
with torch.cuda.stream(background_stream):
    # Préfetch next batch
    pass

with torch.cuda.stream(computation_stream):
    # Compute current batch
    pass
```

---

## 6. Benchmark et Profiling 📊

### Utiliser PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Training step
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### ROCm Profiler

```bash
# Installer rocm-profiler
pip install rocm-profiler

# Profiler l'entraînement
rocprof --stats uv run python train.py
```

---

## 7. Configuration Recommandée 🎯

### Pour RX 7900 XTX (24GB) - Optimisée

```python
# train.py
DEPTH = 6
DEVICE_BATCH_SIZE = 64  # Après gradient checkpointing
TOTAL_BATCH_SIZE = 2**18
WINDOW_PATTERN = "L"

# Activer gradient checkpointing
USE_CHECKPOINT = True

# Mixed precision
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
```

**Performance attendue:**
- MFU: 8-12% (vs 3-5% actuel)
- Tokens/sec: ~250-350K (vs ~150K actuel)
- VRAM utilisée: ~20GB (avec checkpointing)

### Pour MI300X (192GB) - Haute performance

```python
DEPTH = 12
DEVICE_BATCH_SIZE = 128
TOTAL_BATCH_SIZE = 2**20
WINDOW_PATTERN = "SSSL"  # Possible avec plus de mémoire
```

**Performance attendue:**
- MFU: 15-20%
- Tokens/sec: ~1M+

---

## 8. Roadmap des améliorations 📅

### Phase 1 (Immédiat) ✅
- [x] Support ROCm de base
- [x] Gradient checkpointing
- [ ] Flash Attention 2 depuis source

### Phase 2 (Court terme)
- [ ] Triton kernels pour MLP
- [ ] torch.compile avec inductor
- [ ] Data loading optimisé

### Phase 3 (Moyen terme)
- [ ] Flash Attention custom Triton
- [ ] Multi-GPU support
- [ ] Profiling automatique

---

## 9. Dépannage 🆘

### Problème: OOM après optimisations

```python
# Réduire batch size
DEVICE_BATCH_SIZE = 16

# Ou augmenter checkpointing
USE_CHECKPOINT = True  # Sur plus de layers
```

### Problème: Flash Attention ne compile pas

```bash
# Vérifier les dépendances
pip install hipblas hipcub rocsolver

# Réinstaller avec verbose
pip install -v -U flash-attn --no-build-isolation
```

### Problème: torch.compile crash

```python
# Fallback à eager
model = torch.compile(model, backend="eager")
# Ou désactiver
model = model  # Pas de compile
```

---

## 10. Resources 🔗

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/docs/stable/notes/rocm.html)
- [Flash Attention ROCm](https://github.com/ROCm/flash-attention)
- [Triton ROCm](https://triton-lang.org/main/getting-started/installation.html#rocm)
- [AMD Instinct Tuning Guide](https://www.amd.com/content/dam/amd/en/documents/instinct/developer-guides/instinct-tuning-guide.pdf)

---

**Note:** Les gains de performance varient selon le GPU AMD et la version de ROCm. 
Toujours profiler avant/après pour valider les améliorations.
