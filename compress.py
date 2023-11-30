import base64
import gzip
import os
import pickle


model = open("model.pkl", "rb")
# b64_model = base64.b64encode(model.read())
# c_model = gzip.compress(model.read())

with gzip.open("compressed_model_zip.gz", "wb") as f:
    pickle.dump(model.read(), f)

print(os.path.getsize("compressed_model_zip.gz"))

