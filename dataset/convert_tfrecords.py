# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import xml.etree.ElementTree as xml_tree

import numpy as np
import six
import tensorflow as tf

import dataset_common


'''How to organize your dataset folder:
[VOC]
  VOCROOT/
       |->VOC2007/
       |    |->Annotations/
       |    |->ImageSets/
       |    |->...
       |->VOC2012/
       |    |->Annotations/
       |    |->ImageSets/
       |    |->...
       |->VOC2007TEST/
       |    |->Annotations/
       |    |->...
[WIDER_FACE]

|-- WIDERFace
    |-- wider_face_split
    |-- WIDER_train
    |   |-- images [61 entries exceeds filelimit, not opening dir]
    |-- WIDER_val
        |-- images [61 entries exceeds filelimit, not opening dir]

./WIDERFace/wider_face_split/
|-- readme.txt
|-- wider_face_test_filelist.txt
|-- wider_face_test.mat
|-- wider_face_train_bbx_gt.txt
|-- wider_face_train.mat
|-- wider_face_val_bbx_gt.txt
|-- wider_face_val.mat

in wider_face_split, gt_bbox metafiles should be found

'''
tf.app.flags.DEFINE_string('dataset_directory', '/playground/data/WIDERFace/',
                           'All datas directory')
tf.app.flags.DEFINE_string('train_splits', 'WIDER_train',
                           'Comma-separated list of the training data sub-directory')
tf.app.flags.DEFINE_string('validation_splits', 'WIDER_val',
                           'Comma-separated list of the validation data sub-directory')
tf.app.flags.DEFINE_string('train_meta_filename', '/playground/data/WIDERFace/wider_face_split/wider_face_train_bbx_gt.txt',
                           'Train meta filenames [WIDERF]')
tf.app.flags.DEFINE_string('val_meta_filename', '/playground/data/WIDERFace/wider_face_split/wider_face_val_bbx_gt.txt',
                           'Validation meta filenames [WIDERF]')
tf.app.flags.DEFINE_string('output_directory', '/playground/data/WIDERFace/tfrecords',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 8,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')
RANDOM_SEED = 180517

FLAGS = tf.app.flags.FLAGS

# define the wrapper of datatype -> pack into list
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_name, image_buffer, bboxes, labels, labels_text,
                        difficult, truncated, height, width):
    """Build an Example proto for an example.

    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        bboxes: List of bounding boxes for each image
        labels: List of labels for bounding box
        labels_text: List of labels' name for bounding box
        difficult: List of ints indicate the difficulty of that bounding box
        truncated: List of ints indicate the truncation of that bounding box
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        if b[0]>1 or b[1]>1 or b[2]>1 or b[3]>1:
            print(b)
        # pylint: enable=expression-not-assigned
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/shape': _int64_feature([height, width, channels]),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature(labels),
        'image/object/bbox/label_text': _bytes_list_feature(labels_text),
        'image/object/bbox/difficult': _int64_feature(difficult),
        'image/object/bbox/truncated': _int64_feature(truncated),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(filename.encode('utf8')),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file. 
    TODO: adding support to other format of images, including png, jpeg cmyk

      Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
      Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()  # image_data is the string to be returned as encoded string

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)  # decode the data to check the shape of image

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_index_metafiles(filename, base_path):
    """ read the metafiles including the path of image and bbox informations

    Args:
        filename: string; the sub-path of the metafile based on the `base path`
        base_path: string; the base path for the target images

    Returns:
        all_records: list; [img_path,(normalized coords of bbox), cls, cls_name, 0, 0] # fake 0,0 for VOC legacy
    """
    CODER = [1,2,4,8,16,32]
    NAME_CLASS = ['blur', 'exaggrated expression','extreme illum','occlusion','atpyical pose','invalid']

    all_records = []    
    f = open(filename, 'r')
    # parse the metadata_file
    while True:
        line = f.readline().strip('\n')
        if not line:
            break
        filename = base_path+line
        bboxes = []
        ncls = []
        name_cls = []

        n_windows = int(f.readline().strip('\n'))
        for i in range(n_windows):
            cur_window = f.readline().strip('\n').split(' ')
            xmin = int(cur_window[0])
            ymin = int(cur_window[1])
            xmax = xmin + int(cur_window[2])  # ymin + width
            ymax = ymin + int(cur_window[3])  # xmin + height
            cclass = 1
            name_class = ''
            for iec in range(6):
                vec = int(cur_window[iec+4])
                if vec > 0:
                    cclass += 1 * CODER[iec]
                    name_class += NAME_CLASS[iec]
            bboxes.append((ymin,xmin,ymax,xmax))
            ncls.append(cclass)
            name_cls.append(name_class)

        all_records.append([filename, bboxes, ncls, name_cls])

    f.close()
    return all_records


def _find_image_bounding_boxes(directory, cur_record):
    """Find the bounding boxes for *a given image file* - process the `xml` file.

    Args:
        directory: string; the path of all datas.
        cur_record: list of strings; the first of which is the sub-directory of cur_record, the second is the image filename.
    Returns:
        bboxes: List of bounding boxes for each image.
        labels: List of labels for bounding box.
        labels_text: List of labels' name for bounding box.
        difficult: List of ints indicate the difficulty of that bounding box.
        truncated: List of ints indicate the truncation of that bounding box.
    """
    anna_file = os.path.join(directory, cur_record[0], 'Annotations', cur_record[1].replace('jpg', 'xml'))

    tree = xml_tree.parse(anna_file)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(dataset_common.VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        isdifficult = obj.find('difficult')
        if isdifficult is not None:
            difficult.append(int(isdifficult.text))
        else:
            difficult.append(0)

        istruncated = obj.find('truncated')
        if istruncated is not None:
            truncated.append(int(istruncated.text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        # bbox is marked in a normalized manner
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return bboxes, labels, labels_text, difficult, truncated


def _process_image_files_batch_VOC(coder, thread_index, ranges, name, directory, all_records, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: *list of pairs* of integers specifying ranges of each batches to
          analyze in parallel.
        name: string, unique identifier specifying the data set
        directory: string; the path of all datas
        all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
        num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads  # num_threads have to divide num_shards

    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            cur_record = all_records[i]
            filename = os.path.join(directory, cur_record[0], 'JPEGImages', cur_record[1])

            bboxes, labels, labels_text, difficult, truncated = _find_image_bounding_boxes(directory, cur_record)
            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, cur_record[1], image_buffer, bboxes, labels, labels_text,
                                          difficult, truncated, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files_batch_WIDER(coder, thread_index, ranges, name, directory, all_records, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread. (WIDER version)

    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: *list of pairs* of integers specifying ranges of each batches to
          analyze in parallel.
        name: string, unique identifier specifying the data set
        directory: string; the path of all datas
        all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
        num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads  # num_threads have to divide num_shards

    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        # Find the slice of data for current thread
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            cur_record = all_records[i]
            # e.g: [img_file_path, bboxes[(ymin,xmin,ymax,xmax)(ymin,xmin,ymax,xmax)...], n_cls, name_cls]
            filename = cur_record[0]
            bboxes, labels, labels_text = cur_record[1], cur_record[2], cur_record[3]
            image_buffer, height, width = _process_image(filename, coder)
            # normalize the bboxes is a little awkward to go around numpy
            bboxes = np.array(bboxes, dtype=np.float64)
            bboxes[:,0] = bboxes[:,0] / float(height)
            bboxes[:,2] = bboxes[:,2] / float(height)
            bboxes[:,1] = bboxes[:,1] / float(width)
            bboxes[:,3] = bboxes[:,3] / float(width)
            bboxes = bboxes.tolist()

            # example = _convert_to_example(filename, filename, image_buffer, bboxes, 1, 'face',
            #                               0, 0, height, width)
            example = _convert_to_example(filename, filename, image_buffer, bboxes, labels, labels_text,
                                          0, 0, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, directory, all_records, num_shards, raw_db_type='WIDERF'):
    """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    directory: string; the path of all datas
    all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
    num_shards: integer number of shards for this data set.
    raw_db_type: enum('VOC', 'WIDERF'); indicating the type of raw database (support VOC and WIDERF right now) 
  """
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(all_records), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    # Select batch process handle
    if raw_db_type=='VOC':
        batch_process_fn = _process_image_files_batch_VOC
    elif raw_db_type == 'WIDERF':
        batch_process_fn = _process_image_files_batch_WIDER

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, directory, all_records, num_shards)
        t = threading.Thread(target=batch_process_fn, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(all_records)))
    sys.stdout.flush()


def _process_dataset_VOC(name, directory, all_splits, num_shards):
    """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    all_splits: list of strings, sub-path to the data set.
    num_shards: integer number of shards for this data set.
  """
    all_records = []
    for split in all_splits:
        jpeg_file_path = os.path.join(directory, split, 'JPEGImages')
        images = tf.gfile.ListDirectory(jpeg_file_path)  # use tf.gfile list directory
        jpegs = [im_name for im_name in images if im_name.strip()[-3:] == 'jpg']
        all_records.extend(list(zip([split] * len(jpegs), jpegs)))  # all_records is in fact the filenames of the images

    # shuffle the images
    shuffled_index = list(range(len(all_records)))
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled_index)
    all_records = [all_records[i] for i in shuffled_index]
    _process_image_files(name, directory, all_records, num_shards, raw_db_type='VOC')


def _process_dataset_WIDERF(name, directory, all_splits, meta_filenames, num_shards):
    """Process a complete data set and save it as a TFRecord.
    [WIDER Version] in which all data metas(bbox cls) info is stored in a seperate structured file

    Args:
        name: string, unique identifier specifying the data set.
        directory: string, root path to the data set.
        all_splits: list of strings, sub-path to the data set.
        meta_filenames: string; the subpath of the metadata file
        num_shards: integer number of shards for this data set.

    """
    all_records = []

    # check if all_splits is corhenrent with meta-files
    assert len(all_splits) == len(meta_filenames)  
    for split,metadata_file in zip(all_splits, meta_filenames):
        base_path = FLAGS.dataset_directory + split + '/images/'
        all_records.extend(_process_index_metafiles(metadata_file, base_path))
        # attention: here the bbox in the all_records is not normalized
        # all_records e.g: [img_file_path, bboxes[(x,y,width,height)(x,y,width,height)...], n_cls, name_cls]

    _process_image_files(name, directory, all_records, num_shards, raw_db_type='WIDERF')


def parse_comma_list(args):
    return [s.strip() for s in args.split(',')]


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    # Run it!
    # _process_dataset_VOC('val', FLAGS.dataset_directory, parse_comma_list(FLAGS.validation_splits), FLAGS.validation_shards)
    # _process_dataset_VOC('train', FLAGS.dataset_directory, parse_comma_list(FLAGS.train_splits), FLAGS.train_shards)

    _process_dataset_WIDERF('train', FLAGS.dataset_directory, parse_comma_list(FLAGS.train_splits), parse_comma_list(FLAGS.train_meta_filename), FLAGS.train_shards)
    _process_dataset_WIDERF('val', FLAGS.dataset_directory, parse_comma_list(FLAGS.validation_splits), parse_comma_list(FLAGS.val_meta_filename), FLAGS.validation_shards)


if __name__ == '__main__':
    tf.app.run()
