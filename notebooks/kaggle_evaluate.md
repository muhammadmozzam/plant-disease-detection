# Kaggle run (CPU is enough; GPU makes it faster)

Inputs to attach: PlantVillage (mohitsingh1804/plantvillage), PlantDoc
(a Kaggle copy of pratikkayal/PlantDoc-Dataset) and a private dataset
with the three .pth files.

```
!git clone -q https://github.com/muhammadmozzam/plant-disease-detection
!git -C plant-disease-detection log -1 --format=%H
!find /kaggle/input -maxdepth 4 -type d \( -name val -o -iname "*plantdoc*" \) | head
!python plant-disease-detection/backend/evaluate.py \
    --weights /kaggle/input/<models-dataset> \
    --lab /kaggle/input/<plantvillage>/PlantVillage/val \
    --field /kaggle/input/<plantdoc> --out /kaggle/working/results
```
