import sys
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project', '-p', type=str, help='name of the project')
args = parser.parse_args()


def exit (msg):
    print(msg)
    sys.exit()


root_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.join(root_dir, args.project)


config_template = '''
# * this is a template, it does not get used directly *
# this config file gets copied to each run's folder in the /runs directory
# the idea is to be able to tweak specific run's params without affecting others.
# changing things here just means that each time you start a new run, the
# config.yml in that run's directory will be initialized with these settings.
# most likely, you will start a new experiment with the project_skeleton.py,
# and then alter this file once, to set things like the conda envs and library
# locations.

# bash scripts might need to source conda startup script to work with conda
conda_sh: "~/anaconda3/etc/profile.d/conda.sh"

# conda env used to run training (this may be different than what you use to
# run workflow.py, since some utilities may be needed from different versions of
# tensorflow)
conda_env_name: "tf_gpu_14"

# the location of tensorflow's object detection libraries
object_detection_library_dir: "full/path/to/models/research/object_detection/"

# you might need to grab a model and pipeline... so go here if need be:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# and here for pipeline samples, though they are usually included in the model downloads
# https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
base_model_name: "ssd_inception_v2_coco_2018_01_28"

# this will get written in to the pipeline
batch_size: 4


'''


def create_directories ():
    if os.path.exists(proj_dir):
        if 'y' == input('clear existing directories? [y/n] ').lower():
            backup_dir = os.path.join(root_dir, 'backup', args.project)
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(proj_dir, backup_dir)
            shutil.rmtree(proj_dir)
        else:
            exit('will not overwrite old directory. stopping...')
    os.mkdir(os.path.join(proj_dir))
    os.mkdir(os.path.join(proj_dir, 'base_model'))
    os.mkdir(os.path.join(proj_dir, 'detect'))
    os.mkdir(os.path.join(proj_dir, 'images'))
    os.mkdir(os.path.join(proj_dir, 'labels'))
    os.mkdir(os.path.join(proj_dir, 'runs'))
    os.mkdir(os.path.join(proj_dir, 'exports'))
    with open(os.path.join(proj_dir, 'base_config.yml'), 'w') as config_fp:
        config_fp.write(config_template)


if __name__ == '__main__':
    create_directories()
