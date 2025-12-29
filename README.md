# Image Processing Project

## Kurulum

1. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# veya
source .venv/bin/activate  # Linux/Mac
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

**Not:** API değişirse anomalib sürümü pinlenebilir.

## Dataset

MVTecAD veri setini indirip proje dizinine yerleştirin:

**Varsayılan konum:** `./datasets/MVTecAD/`

**Veri seti yapısı:**
```
datasets/
└── MVTecAD/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   │       ├── 000.png
    │   │       └── ...
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   └── ...
    │   └── ground_truth/
    │       └── ...
    ├── cable/
    │   ├── train/
    │   ├── test/
    │   └── ground_truth/
    ├── carpet/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── color/
    │   │   ├── cut/
    │   │   ├── hole/
    │   │   ├── metal_contamination/
    │   │   └── thread/
    │   └── ground_truth/
    └── ... (diğer kategoriler: grid, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper)
```

**Veri setini indir:**
- [MVTecAD resmi sitesi](https://www.mvtec.com/company/research/datasets/mvtec-ad) veya
- Alternatif kaynaklardan indirip `datasets/MVTecAD/` klasörüne çıkarın.

**Özel konum kullanmak için:**
```bash
python scripts/train.py --data_root /path/to/your/dataset --category carpet
```

**Not:** `datasets/` klasörü `.gitignore`'da olduğu için commitlenmeyecektir.

## Train

PatchCore modelini eğitmek için (1 epoch ile hızlı test):

```bash
python scripts/train.py --category carpet --device cuda
```

**Parametreler:**
- `--category`: Kategori adı (varsayılan: `carpet`)
- `--data_root`: Veri seti kök dizini (varsayılan: `./datasets/MVTecAD`)
- `--image_size`: Görüntü boyutu (varsayılan: `256`)
- `--device`: Cihaz (`auto`, `cpu`, `cuda`) - varsayılan: `auto`
- `--output_dir`: Çıktı dizini (varsayılan: `./outputs`)

**Örnek:**
```bash
python scripts/train.py --category bottle --image_size 224 --device cuda
```

Eğitilen model `outputs/models/<run_id>/Patchcore/MVTecAD/<category>/latest/weights/lightning/model.ckpt` konumuna kaydedilir.

## Evaluate

Eğitilmiş modeli test seti üzerinde değerlendirmek ve metrikleri hesaplamak için:

```bash
python scripts/evaluate.py --category carpet --model_path outputs/models/<run_id>/Patchcore/MVTecAD/carpet/latest/weights/lightning/model.ckpt --device cuda
```

**Parametreler:**
- `--category`: Kategori adı (zorunlu)
- `--model_path`: Eğitilmiş model checkpoint yolu (zorunlu)
- `--data_root`: Veri seti kök dizini (varsayılan: `./datasets/MVTecAD`)
- `--image_size`: Görüntü boyutu (varsayılan: `256`)
- `--device`: Cihaz (`auto`, `cpu`, `cuda`) - varsayılan: `auto`
- `--output_dir`: Çıktı dizini (varsayılan: `./outputs`)

**Çıktı:** `outputs/metrics/<run_id>/metrics.json` dosyası oluşturulur (image-level AUROC, kategori, test örnek sayısı vb.).

## Export Samples

Test setinden örnek görselleri anomaly heatmap ve overlay ile birlikte export etmek için:

```bash
python scripts/export_samples.py --category carpet --model_path outputs/models/<run_id>/Patchcore/MVTecAD/carpet/latest/weights/lightning/model.ckpt --num_samples 10 --device cuda
```

**Parametreler:**
- `--category`: Kategori adı (zorunlu)
- `--model_path`: Eğitilmiş model checkpoint yolu (zorunlu)
- `--num_samples`: Export edilecek örnek sayısı (varsayılan: `10`)
- `--random`: Örnekleri rastgele seç (varsayılan: ilk N örnek)
- `--seed`: Rastgele seçim için seed (varsayılan: `42`)
- `--data_root`: Veri seti kök dizini (varsayılan: `./datasets/MVTecAD`)
- `--image_size`: Görüntü boyutu (varsayılan: `256`)
- `--device`: Cihaz (`auto`, `cpu`, `cuda`) - varsayılan: `auto`
- `--output_dir`: Çıktı dizini (varsayılan: `./outputs`)

**Çıktı:** `outputs/samples/<category>/<run_id>/` klasörüne her örnek için 3 görsel kaydedilir:
- `{idx}_label{pred_label}_score{score:.3f}_input.png` - Orijinal görüntü
- `{idx}_label{pred_label}_score{score:.3f}_heatmap.png` - Anomaly heatmap
- `{idx}_label{pred_label}_score{score:.3f}_overlay.png` - Overlay (görüntü + heatmap)

## Run UI

Streamlit arayüzünü çalıştırmak için:

```bash
streamlit run app.py
```

## Rapor için Çıktı Nerede?

### Metrikler

Değerlendirme sonuçları:
```
outputs/metrics/<run_id>/metrics.json
```

İçerik örneği:
```json
{
  "category": "carpet",
  "n_test": 117,
  "device": "cuda",
  "image_auroc": 0.9234
}
```

### Görsel Örnekler

Export edilen görseller:
```
outputs/samples/<category>/<run_id>/
├── 0000_label0_score0.695_input.png
├── 0000_label0_score0.695_heatmap.png
├── 0000_label0_score0.695_overlay.png
├── 0001_label0_score0.705_input.png
└── ...
```

**Not:** `outputs/` klasörü `.gitignore`'da olduğu için commitlenmeyecektir. Rapor için gerekli dosyaları manuel olarak kopyalayın.

