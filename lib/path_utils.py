from shutil import rmtree
import json
import os

def save_model_and_settings(model, settings, verbose=0):
    out_dir = settings['output']['folder']
    if os.path.exists(out_dir):
        rmtree(out_dir)
    os.mkdir(out_dir)

    model_path = os.path.join(out_dir, 'word2vec.model')
    settings_path = os.path.join(out_dir, 'settings.json')

    model.save(model_path)
    if verbose:
        print(f'saved model to {model_path}')
    with open(settings_path, 'w') as outfile:
        json.dump(settings, outfile)