"""PlantDoc folder name -> matching PlantVillage class.

PlantDoc (Singh et al., 2020) has real field photos of the same crops.
Only classes with a clear one-to-one match are used; anything else is left
out and listed in the results, so the comparison stays fair.
"""

PLANTDOC_TO_PLANTVILLAGE = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple leaf": "Apple___healthy",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "bell_pepper leaf": "Pepper,_bell___healthy",
    "bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust leaf": "Corn_(maize)___Common_rust_",
    "peach leaf": "Peach___healthy",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def _key(name):
    # folder names come as "Apple Scab Leaf", "Apple_Scab_Leaf" or "apple-scab-leaf"
    return " ".join(name.strip().lower().replace("-", " ").replace("_", " ").split())


_LOOKUP = {_key(k): v for k, v in PLANTDOC_TO_PLANTVILLAGE.items()}


def match_plantdoc_folder(folder_name):
    return _LOOKUP.get(_key(folder_name))
