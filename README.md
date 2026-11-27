# Plant Disease Detection: from lab photos to field photos

Three image classifiers (ResNet18, a small custom CNN and MobileNetV2) trained to recognise 38 plant diseases and healthy leaves, served through a FastAPI backend and a React web app. The current work measures how well these lab-trained models hold up on real photos taken in fields.

## Project history

- **March–May 2026:** built offline. Trained the three models on PlantVillage, wrote the FastAPI prediction API and the React + Vite frontend with a model selector.
- **November 2026:** released the code and added a measured evaluation: a lab test on the PlantVillage validation split, and a field test on PlantDoc photos of the same crops and diseases.

## The question

PlantVillage photos show a single leaf on a plain background in controlled light, and models trained on them often score above 95%. Farmers' photos look different: several leaves, soil, shadows, hands, different cameras. This project asks how much accuracy each model keeps when it moves from the lab to the field, and whether the smaller, faster models lose more.

## Data

| Dataset | Use | Images | Notes |
|---|---|---|---|
| [PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) (Hughes & Salathé, 2015) | Training (80%) and lab test (20%) | 54,305 colour images, 38 classes | Fixed train/val folders; the 20% val folder was never used for training |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (Singh et al., 2020) | Field test only | Field photos of 13 crops | Only the classes that match a PlantVillage class are used ([mapping](backend/plantdoc_map.py)); the rest are listed in the results |

Neither dataset is included in this repository.

## Models

| Model | Training | Parameters |
|---|---|---|
| ResNet18 | ImageNet-pretrained, fine-tuned 5 epochs, Adam (lr 0.001), batch 32 | 11.2 M |
| Custom CNN | 3 conv blocks + 2 dense layers, trained from scratch, 5 epochs | 6.5 M |
| MobileNetV2 | ImageNet-pretrained, fine-tuned 5 epochs | 2.3 M |

All models use 224×224 inputs. Training scripts are in `backend/` (`rn_train_model.py`, `train_cnn.py`, `train_mobilenet.py`).

## Results

Measured with [`backend/evaluate.py`](backend/evaluate.py): accuracy, macro-F1, top-3 accuracy, per-class reports, confusion matrices, CPU time per image and model size, on the lab test and the field test.

Measured on 27 Nov 2026 on Kaggle (CPU). The field test uses all 2,922 PlantDoc photos, each mapped to one of 28 shared classes; random guessing would score about 3.6%.

| Model | Lab accuracy (PlantVillage val, 38 classes) | Lab accuracy (28 shared classes) | **Field accuracy (PlantDoc, 28 shared classes)** | Field top-3 | Field macro-F1 | CPU time / image | Size |
|---|---|---|---|---|---|---|---|
| ResNet18 | 97.7% | 97.7% | **13.3%** | 30.6% | 0.10 | 46.6 ms | 42.8 MB |
| Custom CNN | 91.5% | 92.5% | **5.4%** | 15.1% | 0.04 | 9.8 ms | 24.6 MB |
| MobileNetV2 | 99.0% | 98.9% | **10.0%** | 25.7% | 0.09 | 23.8 ms | 8.9 MB |

**What this shows**

- All three models look nearly perfect on lab photos and fail on field photos: accuracy falls by 85–89 percentage points.
- The custom CNN trained from scratch is close to random guessing in the field (5.4% vs 3.6%); the two ImageNet-pretrained models keep a little more (10–13%).
- The best lab model (MobileNetV2) is not the best field model (ResNet18), so lab accuracy alone is a poor guide for choosing a model.

Per-class reports and confusion matrices are produced by the same script.

> The charts made by `compare_models.py` in the original project used hand-typed numbers and rough training times, so they are not used as results.

## Run it

**Evaluation** (on Kaggle or any machine with the datasets):

```bash
pip install -r requirements.txt
python backend/evaluate.py --weights path/to/model_files \
    --lab path/to/PlantVillage/val --field path/to/PlantDoc --out results
```

**Web app**

```bash
# backend: put the three .pth files in backend/
cd backend
uvicorn api:app --reload          # http://127.0.0.1:8000/docs

# frontend
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

The trained model files are not stored in git because of their size; a download link will be added here.

## Repository layout

```
backend/
  api.py               FastAPI: POST /predict (image + model_name)
  models.py            the three architectures, defined once
  evaluate.py          lab and field evaluation
  plantdoc_map.py      PlantDoc -> PlantVillage class mapping
  classes.json         class order used by all models
  rn_train_model.py, train_cnn.py, train_mobilenet.py   training
  dataset_analysis.py  class counts and sample images
  figures/             training curves from the original runs
frontend/              React + Vite app
```

## Limitations

- A research prototype for learning and comparison, not a tool for real crop decisions.
- PlantDoc covers only some PlantVillage classes, so the field test uses the shared classes.
- Each model was trained once (5 epochs, no fixed random seed), so small differences between models may not be meaningful.

## References

- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.
- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A dataset for visual plant disease detection. *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 249–253.

## Author

Muhammad Mozzam · [GitHub](https://github.com/muhammadmozzam)

Licence: MIT
