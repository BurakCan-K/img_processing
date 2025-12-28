# Image Processing Project

## Kurulum

**Not:** API değişirse anomalib sürümü pinlenebilir.

## Dataset

MVTecAD veri setini indirip proje dizinine yerleştirin:

**Varsayılan konum:** `./datasets/MVTecAD/`

Veri seti yapısı şu şekilde olmalı:
```
datasets/
└── MVTecAD/
    ├── bottle/
    │   ├── train/
    │   └── test/
    ├── cable/
    │   ├── train/
    │   └── test/
    ├── carpet/
    │   ├── train/
    │   └── test/
    └── ... (diğer kategoriler)
```

**Veri setini indir:**
- [MVTecAD resmi sitesi](https://www.mvtec.com/company/research/datasets/mvtec-ad) veya
- Alternatif kaynaklardan indirip `datasets/MVTecAD/` klasörüne çıkarın.

**Özel konum kullanmak için:**
```bash
python scripts/train.py --data_root /path/to/your/dataset --category carpet
```

## Train

PatchCore modelini eğitmek için:

```bash
python scripts/train.py --category carpet
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

Eğitilen model `outputs/models/<run_id>/model.ckpt` konumuna kaydedilir.

## Evaluate

## Export Samples

## Run UI

