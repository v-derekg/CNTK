
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import pytest

from cntk.io import *
import cntk.io.transforms as xforms

abs_path = os.path.dirname(os.path.abspath(__file__))

AA = np.asarray

def test_text_format(tmpdir):
    mbdata = r'''0	|x 560:1	|y 1 0 0 0 0
0	|x 0:1
0	|x 0:1
1	|x 560:1	|y 0 1 0 0 0
1	|x 0:1
1	|x 0:1
1	|x 424:1
'''
    tmpfile = str(tmpdir/'mbdata.txt')
    with open(tmpfile, 'w') as f:
        f.write(mbdata)

    input_dim = 1000
    num_output_classes = 5

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
         features  = StreamDef(field='x', shape=input_dim, is_sparse=True),
         labels    = StreamDef(field='y', shape=num_output_classes, is_sparse=False)
       )))

    assert isinstance(mb_source, MinibatchSource)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.next_minibatch(7)

    features = mb[features_si]
    # 2 samples, max seq len 4, 1000 dim
    assert features.shape == (2, 4, input_dim)
    assert features.end_of_sweep
    assert features.num_sequences == 2
    assert features.num_samples == 7
    assert features.is_sparse
    # TODO features is sparse and cannot be accessed right now:
    # *** RuntimeError: DataBuffer/WritableDataBuffer methods can only be called for NDArrayiew objects with dense storage format
    # 2 samples, max seq len 4, 1000 dim
    #assert features.data().shape().dimensions() == (2, 4, input_dim)
    #assert features.data().is_sparse()

    labels = mb[labels_si]
    # 2 samples, max seq len 1, 5 dim
    assert labels.shape == (2, 1, num_output_classes)
    assert labels.end_of_sweep
    assert labels.num_sequences == 2
    assert labels.num_samples == 2
    assert not labels.is_sparse

    label_data = np.asarray(labels)
    assert np.allclose(label_data,
            np.asarray([
                [[ 1.,  0.,  0.,  0.,  0.]],
                [[ 0.,  1.,  0.,  0.,  0.]]
                ]))

    mb = mb_source.next_minibatch(1)
    features = mb[features_si]
    labels = mb[labels_si]

    assert not features.end_of_sweep
    assert not labels.end_of_sweep
    assert features.num_samples < 7
    assert labels.num_samples == 1

def test_sweeps_and_samples_in_minibatch_source_config():
    config = MinibatchSourceConfig(False)

    assert INFINITELY_REPEAT == config.get_max_samples()
    assert INFINITELY_REPEAT == config.get_max_sweeps()

    config.set_max_samples(100)
    assert 100 == config.get_max_samples()
    assert INFINITELY_REPEAT == config.get_max_sweeps()

    config.set_max_samples(20)
    assert 20 == config.get_max_samples()
    assert INFINITELY_REPEAT == config.get_max_sweeps()

    config.set_max_sweeps(3)
    assert 20 == config.get_max_samples()
    assert 3 == config.get_max_sweeps()

    dictionary = config.as_dictionary()

    assert 3 == len(dictionary.keys())
    assert 20 == dictionary['maxSamples']
    assert 3 == dictionary['maxSweeps']
    assert False == dictionary['randomize']
    
def test_randomization_in_minibatch_source_config():
    config = MinibatchSourceConfig(True)
    dictionary = config.as_dictionary()

    assert 3 == len(dictionary.keys())
    assert False == dictionary['sampleBasedRandomizationWindow']
    assert DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS == dictionary['randomizationWindow']
    assert True == dictionary['randomize']

    config.set_randomization_window_in_samples(12345)
    dictionary = config.as_dictionary()

    assert 3 == len(dictionary.keys())
    assert True == dictionary['sampleBasedRandomizationWindow']
    assert 12345 == dictionary['randomizationWindow']
    assert True == dictionary['randomize']

    config.set_randomization_window_in_chunks(321)
    dictionary = config.as_dictionary()

    assert 3 == len(dictionary.keys())
    assert False == dictionary['sampleBasedRandomizationWindow']
    assert 321 == dictionary['randomizationWindow']
    assert True == dictionary['randomize']

def test_all_other_parameters_minibatch_source_config():
    config = MinibatchSourceConfig(False) \
        .set_trace_level(TraceLevel.Warning) \
        .set_frame_mode(True) \
        .set_multithreaded(True)


    dictionary = config.as_dictionary()

    assert 4 == len(dictionary.keys())
    assert 1 == dictionary['traceLevel']
    assert True == dictionary['frameMode']
    assert True == dictionary['multiThreadedDeserialization']
    
    config.set_frame_mode(False).set_multithreaded(False) \
          .set_trace_level(4) \
          .set_truncation_length(314)

    assert 6 == len(dictionary.keys())
    assert 4 == dictionary['traceLevel']
    assert False == dictionary['frameMode']
    assert False == dictionary['multiThreadedDeserialization']
    assert True == dictionary['truncated']
    assert 314 == dictionary['truncationLength']

def test_image():
    map_file = "input.txt"
    mean_file = "mean.txt"
    epoch_size = 150

    feature_name = "f"
    image_width = 100
    image_height = 200
    num_channels = 3

    label_name = "l"
    num_classes = 7

    transforms = [xforms.crop(crop_type='randomside', side_ratio=0.5, jitter_type='uniratio'),
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)]
    image = ImageDeserializer(map_file, StreamDefs(f = StreamDef(field='image', transforms=transforms), l = StreamDef(field='label', shape=num_classes)))

    config = MinibatchSourceConfig(randomize=False).set_max_samples(epoch_size).add_deserializer(image)
    rc = config.as_dictionary()

    assert rc['maxSamples'] == epoch_size
    assert rc['randomize'] == False

    assert len(rc['deserializers']) == 1
    d = rc['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    f = d['input'][feature_name]
    assert set(f.keys()) == { 'transforms' }
    t0, t1, t2 = f['transforms']
    assert t0['type'] == 'Crop'
    assert t1['type'] == 'Scale'
    assert t2['type'] == 'Mean'
    assert t0['cropType'] == 'randomside'
    assert t0['sideRatio'] == 0.5
    assert t0['aspectRatio'] == 1.0
    assert t0['jitterType'] == 'uniratio'
    assert t1['width'] == image_width
    assert t1['height'] == image_height
    assert t1['channels'] == num_channels
    assert t1['interpolations'] == 'linear'
    assert t2['meanFile'] == mean_file

    randomization_window = 100

    config = MinibatchSourceConfig(randomize=False).set_max_samples(epoch_size).add_deserializer(image) \
                .set_randomization_window_in_samples(randomization_window)
    rc = config.as_dictionary()

    assert rc['maxSamples'] == epoch_size
    assert rc['randomize'] == True
    assert rc['randomizationWindow'] == randomization_window
    assert rc['sampleBasedRandomizationWindow'] == True
    assert len(rc['deserializers']) == 1
    d = rc['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    config = MinibatchSourceConfig().set_max_samples(FULL_DATA_SWEEP).add_deserializer(image) \
                .set_randomization_window_in_chunks(randomization_window)
    rc = config.as_dictionary()

    assert rc['maxSamples'] == FULL_DATA_SWEEP
    assert rc['randomize'] == True
    assert rc['randomizationWindow'] == randomization_window
    assert rc['sampleBasedRandomizationWindow'] == False
    assert len(rc['deserializers']) == 1
    d = rc['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    # TODO depends on ImageReader.dll
    '''
    mbs = rc.minibatch_source()
    sis = mbs.stream_infos()
    assert set(sis.keys()) == { feature_name, label_name }
    '''

def test_full_sweep_minibatch(tmpdir):

    mbdata = r'''0	|S0 0   |S1 0
0	|S0 1 	|S1 1
0	|S0 2
0	|S0 3 	|S1 3
1	|S0 4
1	|S0 5 	|S1 1
1	|S0 6	|S1 2
'''

    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write(mbdata)

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
        features  = StreamDef(field='S0', shape=1),
        labels    = StreamDef(field='S1', shape=1))),
        randomize=False, epoch_size=FULL_DATA_SWEEP)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.next_minibatch(1000)

    assert mb[features_si].num_sequences == 2
    assert mb[labels_si].num_sequences == 2

    features = mb[features_si]
    assert features.end_of_sweep
    assert len(features.value) == 2
    expected_features = \
            [
                [[0],[1],[2],[3]],
                [[4],[5],[6]]
            ]

    for res, exp in zip (features.value, expected_features):
        assert np.allclose(res, exp)

    assert np.allclose(features.mask,
            [[2, 1, 1, 1],
             [2, 1, 1, 0]])

    labels = mb[labels_si]
    assert labels.end_of_sweep
    assert len(labels.value) == 2
    expected_labels = \
            [
                [[0],[1],[3]],
                [[1],[2]]
            ]
    for res, exp in zip (labels.value, expected_labels):
        assert np.allclose(res, exp)

    assert np.allclose(labels.mask,
            [[2, 1, 1],
             [2, 1, 0]])

def create_temp_file(tmpdir):
    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write("|S0 1\n|S0 2\n|S0 3\n|S0 4")
    return tmpfile

def create_config(tmpdir):
    tmpfile = create_temp_file(tmpdir)
    return MinibatchSourceConfig() \
        .add_deserializer(
            CTFDeserializer(tmpfile, 
                StreamDefs(features  = StreamDef(field='S0', shape=1))))

def test_set_max_samples(tmpdir):
    mb_source = create_config(tmpdir) \
        .set_max_samples(1) \
        .create_minibatch_source()

    input_map = {'features' : mb_source['features']}
    mb = mb_source.next_minibatch(10, input_map)

    assert 'features' in mb
    assert mb['features'].num_samples == 1
    assert not mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(10, input_map)

    assert not mb

def test_set_max_sweeps(tmpdir):
    # set max sweeps to 3 (12 samples altogether).
    mb_source = create_config(tmpdir) \
        .set_max_sweeps(3) \
        .create_minibatch_source()

    input_map = {'features' : mb_source['features']}

    for i in range(2):
        mb = mb_source.next_minibatch(5, input_map)

        assert 'features' in mb
        assert mb['features'].num_samples == 5
        assert mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(5, input_map)

    assert 'features' in mb
    assert mb['features'].num_samples == 2
    assert mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(1, input_map)

    assert not mb

def test_set_max_sweeps_and_max_samples(tmpdir):
    # set max sweeps to 3 (12 samples altogether).
    mb_source = create_config(tmpdir) \
        .set_max_sweeps(3) \
        .set_max_samples(11) \
        .create_minibatch_source()

    input_map = {'features' : mb_source['features']}

    for i in range(2):
        mb = mb_source.next_minibatch(5, input_map)

        assert 'features' in mb
        assert mb['features'].num_samples == 5
        assert mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(5, input_map)

    assert 'features' in mb
    assert mb['features'].num_samples == 1
    assert not mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(1, input_map)

    assert not mb

def test_one_sweep(tmpdir):
    sources = []
    sources.append(create_config(tmpdir) \
        .set_max_sweeps(1) \
        .create_minibatch_source())

    sources.append(create_config(tmpdir) \
        .set_max_sweeps(INFINITELY_REPEAT) \
        .set_max_samples(FULL_DATA_SWEEP) \
        .create_minibatch_source())

    sources.append(create_config(tmpdir) \
        .set_max_sweeps(1) \
        .set_max_samples(FULL_DATA_SWEEP) \
        .create_minibatch_source())

    sources.append(create_config(tmpdir) \
        .set_max_sweeps(1) \
        .set_max_samples(INFINITELY_REPEAT) \
        .create_minibatch_source())

    for source in sources:
        input_map = {'features' : source['features']}

        mb = source.next_minibatch(100, input_map)

        assert 'features' in mb
        assert mb['features'].num_samples == 4
        assert mb['features'].end_of_sweep

        mb = source.next_minibatch(100, input_map)

        assert not mb

def test_large_minibatch(tmpdir):

    mbdata = r'''0  |S0 0   |S1 0
0   |S0 1   |S1 1
0   |S0 2
0   |S0 3   |S1 3
0   |S0 4
0   |S0 5   |S1 1
0   |S0 6   |S1 2
'''

    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write(mbdata)

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
        features  = StreamDef(field='S0', shape=1),
        labels    = StreamDef(field='S1', shape=1))),
        randomize=False)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.next_minibatch(1000)
    features = mb[features_si]
    labels = mb[labels_si]

    # Actually, the minibatch spans over multiple sweeps,
    # not sure if this is an artificial situation, but
    # maybe instead of a boolean flag we should indicate
    # the largest sweep index the data was taken from.
    assert features.end_of_sweep
    assert labels.end_of_sweep

    assert features.num_samples == 1000 - 1000 % 7
    assert labels.num_samples == 5 * (1000 // 7)

    assert mb[features_si].num_sequences == (1000 // 7)
    assert mb[labels_si].num_sequences == (1000 // 7)


@pytest.mark.parametrize("idx, alias_tensor_map, expected", [
    (0, {'A': [object()]}, ValueError),
])
def test_sequence_conversion_exceptions(idx, alias_tensor_map, expected):
    with pytest.raises(expected):
        sequence_to_cntk_text_format(idx, alias_tensor_map)


@pytest.mark.parametrize("idx, alias_tensor_map, expected", [
    (0, {'W': AA([])}, ""),
    (0, {'W': AA([[[1, 0, 0, 0], [1, 0, 0, 0]]])}, """\
0\t|W 1 0 0 0 1 0 0 0\
"""),
    (0, {
        'W': AA([[[1, 0, 0, 0], [1, 0, 0, 0]]]),
        'L': AA([[[2]]])
    },
        """\
0\t|L 2 |W 1 0 0 0 1 0 0 0\
"""),
    (0, {
        'W': AA([[[1, 0], [1, 0]], [[5, 6], [7, 8]]]),
        'L': AA([[[2]]])
    },
        """\
0\t|L 2 |W 1 0 1 0
0\t|W 5 6 7 8"""),
])
def test_sequence_conversion_dense(idx, alias_tensor_map, expected):
    assert sequence_to_cntk_text_format(idx, alias_tensor_map) == expected


@pytest.mark.parametrize("data, expected", [
    ([1], True),
    ([[1, 2]], True),
    ([[AA([1, 2])]], False),
    ([AA([1, 2])], False),
    ([AA([1, 2]), AA([])], False),
])
def test_is_tensor(data, expected):
    from cntk.io import _is_tensor
    assert _is_tensor(data) == expected
