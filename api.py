from argparse import ArgumentParser
from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def predict():
    isinput_complete, input_vals = parse_request_input()
    if not isinput_complete:
        return jsonify({'error': ["Parameters list is wrong. The model expects the following variables: {}".format(",".join(predictors_names))]})
    else:
        isvalid, valid_input = isvalid_input(input_vals)
        if not isvalid:
            return jsonify({'error': ["Params values are not valid."]})
        else:
            x = np.array(list(valid_input.values()))
            x = x.reshape(1, -1)
            x = normalizer.transform(x)
            x = transformer.transform(x)
            prediction = best_model_instance.predict(x)
            return jsonify({'prediction': [list(prediction)[0]]})

def isvalid_input(input_vals):
    try:
        validated = {k:float(v) for k,v in input_vals.items()}
        for k,v in validated.items():
            assert v >=0
        return True, validated
    except:
        return False, None

def parse_request_input():
    try:
        temp = {k:request.args.get(k) for k in predictors_names if request.args.get(k) is not None}
        assert set(list(temp.keys())) == set(predictors_names)
        return True, temp
    except:
        return False, {}

def args_parser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--folder", dest="folder", default="./assets",
                        help="Folder containing best model resources (checkpoint, normalizer, transformer if any)")
    parser.add_argument("-n", "--noramlizer",
                        dest="normalizer", default="normalizer.save",
                        help="normalizer config file.")
    parser.add_argument("-t", "--transformer", default="transformer.save",
                        dest="transformer", help="transformer config file, if any.")
    parser.add_argument("-c", "--checkpoint",
                        dest="checkpoint", default="model.save",
                        help="checkpoint config file.")
    parser.add_argument("-g", "--config",
                        dest="config", default="config.save",
                        help="other config file.")
    parser.add_argument("-p", "--predictors",
                        dest="predictors", default="features.save",
                        help="predictors names file.")
    parser.add_argument("-o", "--port",
                        dest="port", default=8080,
                        help="predictors names file.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    try:
        normalizer = joblib.load(args.folder+'/'+args.normalizer) 
        transformer = joblib.load(args.folder+'/'+args.transformer) 
        best_model_instance = joblib.load(args.folder+'/'+args.checkpoint) 
        predictors_names = joblib.load(args.folder+'/'+args.predictors) 
        print("API started successfully on http://127.0.0.1:{}".format(args.port))
        print("Predict using http://127.0.0.1:{}/predict".format(args.port))
        app.run(port=args.port)
    except:
        print("Error loading necessary files {},{}, {} and {} from {}. API won't be launched. Please check again that the given folder and files do exist and are valid ones.".format(args.normalizer, args.config, args.transformer, args.checkpoint, args.folder))