import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_test_cases(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['test_cases']
