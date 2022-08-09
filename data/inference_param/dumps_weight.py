import pickle
import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)
print(json.dumps(network, cls=NumpyArrayEncoder))