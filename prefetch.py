from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  

model = DeepFace.build_model("SFace")

print("✅ SFace model preloaded successfully.")
