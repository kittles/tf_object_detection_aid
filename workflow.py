from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import os
import shutil
import glob
import re
import io
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--project', '-p', type=str, help='name of the project')
args = parser.parse_args()

root_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.join(root_dir, args.project)
runs_dir = os.path.join(proj_dir, 'runs')

config_fp = os.path.join(proj_dir, 'base_config.yml')
config = yaml.load(open(config_fp, 'r'), Loader=yaml.FullLoader)

def exit (msg):
    print(msg)
    sys.exit()


def run_dir_name ():
    if len(os.listdir(runs_dir)):
        run_num = 1 + (max([int(i) for i in os.listdir(runs_dir)]))
    else:
        run_num = 1
    return '{:03}'.format(run_num)


current_run_dir = os.path.join(runs_dir, run_dir_name())
os.mkdir(current_run_dir)
print('setting up new run in {}'.format(current_run_dir))

# labels to csv

classes = [] # this is used later for label_map.pbtxt generation
label_fp = os.path.join(proj_dir, 'labels')
label_csv_fp = os.path.join(current_run_dir, 'labels.csv')
xml_list = []
if len(glob.glob(label_fp + '/*.xml')) == 0:
    exit('no labels found, stopping...')
for xml_file in glob.glob(label_fp + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (
            root.find('path').text,
            int(root.find('size')[0].text),
            int(root.find('size')[1].text),
            member[0].text,
            int(member[4][0].text),
            int(member[4][1].text),
            int(member[4][2].text),
            int(member[4][3].text)
        )
        classes.append(member[0].text)
        xml_list.append(value)
column_name = [
    'filename', 
    'width', 
    'height',
    'class', 
    'xmin', 
    'ymin', 
    'xmax',
    'ymax',
]
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv(label_csv_fp, index=None)
classes = list(set(classes))


# split in to train and test labels

np.random.seed(1)
full_labels = pd.read_csv(label_csv_fp)
grouped = full_labels.groupby('filename')
gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]
n = len(grouped_list)
train_index = np.random.choice(len(grouped_list), size=n-(int(n/4)), replace=False)
test_index = np.setdiff1d(list(range(n)), train_index)
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])
num_examples = {
    'train': len(train),
    'test': len(test),
}
dataset_info = 'instances:\ntrain count: {}\ntest count: {}\ntotal: {}\n'.format(len(train), len(test), len(train) + len(test))
dataset_info = 'images:\ntrain count: {}\ntest count: {}\ntotal: {}'.format(len(train), len(test), len(train) + len(test))
with open(os.path.join(current_run_dir, 'dataset_info.txt'), 'w') as fh:
    fh.write(dataset_info)
train_csv_fp = os.path.join(current_run_dir, 'train_labels.csv')
test_csv_fp = os.path.join(current_run_dir, 'test_labels.csv')
train.to_csv(train_csv_fp, index=None)
test.to_csv(test_csv_fp, index=None)


# generate label_map.pbtxt

def to_item_string (idx, c):
    s = 'item {\n'
    s += '  id: {}\n'.format(idx)
    s += '  name: \'{}\'\n'.format(c)
    s += '}\n'
    return s


label_map_fp = os.path.join(current_run_dir, 'label_map.pbtxt')
with open(label_map_fp, 'w') as fp:
    for idx, c in enumerate(classes):
        fp.write(to_item_string(idx + 1, c))
labels = ['offset'] + classes # for getting class number


# generate tfrecords (this is adapted from some script somewhere...)

csv_input_fps = [train_csv_fp, test_csv_fp]
train_tf_fp = os.path.join(current_run_dir, 'train.tfrecord')
test_tf_fp = os.path.join(current_run_dir, 'test.tfrecord')
tf_output_fps = [train_tf_fp, test_tf_fp]
image_dir = os.path.join(proj_dir, 'images')

def class_text_to_int (row_label):
    label_id = labels.index(row_label)
    if label_id == -1:
        # should raise error
        return None
    else:
        return label_id

def split (df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example (group):
    with tf.compat.v1.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
for csv_input_fp, tf_output_fp in zip(csv_input_fps, tf_output_fps):
    writer = tf.compat.v1.python_io.TFRecordWriter(tf_output_fp)
    examples = pd.read_csv(csv_input_fp)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())
    writer.close()
    output_path = os.path.join(os.getcwd(), tf_output_fp)


pipeline_template_fp = os.path.join(proj_dir, 'base_model', config['base_model_name'], 'pipeline.config')
pipeline_fp = os.path.join(current_run_dir, 'pipeline.config')
with open(pipeline_fp, 'w') as dest:
    with open(pipeline_template_fp, 'r') as fh:
        for line in fh:
            if 'batch_size:' in line:
                line = line.split(':')[0] + ': {}\n'.format(config['batch_size'])
            if 'num_classes:' in line:
                line = line.split(':')[0] + ': {}\n'.format(len(classes))
            if 'num_examples:' in line:
                line = line.split(':')[0] + ': {}\n'.format(num_examples['test'])
            if 'label_map_path:' in line:
                line = line.split(':')[0] + ': "{}"\n'.format(label_map_fp)
            if 'train.record' in line:
                line = line.split(':')[0] + ': "{}"\n'.format(train_tf_fp)
            if 'val.record' in line:
                line = line.split(':')[0] + ': "{}"\n'.format(test_tf_fp)
            if 'fine_tune_checkpoint:' in line:
                line = line.split(':')[0] + ': "{}"\n'.format(os.path.join(proj_dir, 'base_model', config['base_model_name'], 'model.ckpt'))
            dest.write(line)

training_script_text = '''\
cd {research_dir}
source {conda_sh}
conda activate {conda_env_name}
python object_detection/model_main.py \
--pipeline_config_path={pipeline_config_path} \
--model_dir={model_dir} \
--num_train_steps={num_train_steps} \
--sample_1_of_n_eval_examples={sample_1_of_n_eval_examples} \
--alsologtostderr
'''.format(**{
    # these should probably come from a config.yml actually
    'research_dir': config['object_detection_library_dir'],
    'conda_sh': config['conda_sh'],
    'conda_env_name': config['conda_env_name'],
    'pipeline_config_path': os.path.join(current_run_dir, 'pipeline.config'),
    'model_dir': current_run_dir,
    'num_train_steps': 200000,
    'sample_1_of_n_eval_examples': 1,
})
training_script_fp = os.path.join(current_run_dir, 'run_training.sh')
with open(training_script_fp, 'w') as fh:
    fh.write(training_script_text)
os.chmod(training_script_fp, 0o775)


export_script_text = '''\
cd {research_dir}/object_detection
source {conda_sh}
conda activate {conda_env_name}
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path {pipeline_config_path} \
--trained_checkpoint_prefix $(something to get highest model checkpoint) \
--output_directory {exports_dir}
'''.format(**{
    # these should probably come from a config.yml actually
    'research_dir': config['object_detection_library_dir'],
    'conda_sh': config['conda_sh'],
    'conda_env_name': config['conda_env_name'],
    'pipeline_config_path': os.path.join(current_run_dir, 'pipeline.config'),
    'exports_dir': os.path.join(current_run_dir, 'exports'),
})
training_script_fp = os.path.join(current_run_dir, 'export_model.sh')
with open(training_script_fp, 'w') as fh:
    fh.write(export_script_text)
os.chmod(training_script_fp, 0o775)
