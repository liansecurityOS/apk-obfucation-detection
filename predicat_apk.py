from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.androconf import show_logging
from lstm import predict
import sys
import numpy as np
import time

def get_write_content(c):
    class_name = c.get_name()

    class_name = class_name.split('/')[-1]
    class_name = class_name.replace("$", ' ')
    text = "Class: " + class_name + " " + "Method: " + ' '.join([m.get_name() for m in c.get_methods()]) + " " + "Field: " + ' '.join([f.get_name() for f in c.get_fields()])
    return text

def extract_apk_features(path):
    """
    Get a list of custom properties
    :param path: the path to the
    :rtype: string
    """
    a = APK(path)
    d = DalvikVMFormat(a)
    results = []
    for c in d.get_classes():
        #print(c.get_name())
        text = get_write_content(c)
        results.append(text)
    return results

def predict_apk(apk_path):
    apk_features = extract_apk_features(apk_path)
    binary_predictions = predict(apk_features)
    #Coverage of APK obfuscation
    count = np.count_nonzero(binary_predictions == 1)
    obfuscation_coverage = count/len(apk_features)
    return obfuscation_coverage

if __name__ == "__main__":
    apk_path = sys.argv[1]
    obfuscation_coverage = predict_apk(apk_path)
    print("Obfuscation_coverage: ", obfuscation_coverage)