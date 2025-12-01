from deepface import DeepFace
import pandas as pd
import os

DB_PATH ="Recon_Pool"

TARGET_IMG = "targets/Aaron_Peirsol_0001.jpg"

MODEL_NAME= "ArcFace" # ArcFace = 512 Vectors (Supreme), FaceNet = 128 Vectors (Great results),

if not os.path.isdir(DB_PATH):
    print(f"Oh no there is no file in there:{DB_PATH} WHAT TO DO? OH GOD NOOO!!!")

else:

    try:
        results_list = DeepFace.find(
            img_path = TARGET_IMG,
            db_path = DB_PATH,
            model_name = MODEL_NAME,
            detector_backend = 'retinaface', # Face DEtection:(retinaface, opencv, ssd ) Finds the face in the picture to send it to the Model (ArcFace)
            distance_metric = 'cosine', # Metrics for Embedding Vectors: cosine (best), euclidean & euclidean_12 
            enforce_detection = True # Error checking-> If there is no face, retinaface will stop 

        )

        df = results_list[0]

        if df.empty:
            print("❌ No matches hunny" )
        else:
            print(df)



    except Exception as e:
        print(f"\n ❌ ERROR: {e}")
