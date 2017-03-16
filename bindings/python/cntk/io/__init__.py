# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py
from ..tensor import ArrayMixin
from ..utils import typemap, value_to_seq
from cntk.device import use_default_device
from enum import Enum, unique

import numpy as np

INFINITELY_REPEAT = cntk_py.MinibatchSource.infinitely_repeat
FULL_DATA_SWEEP = cntk_py.MinibatchSource.full_data_sweep
DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS = cntk_py.MinibatchSource.default_randomization_window_in_chunks

class MinibatchData(cntk_py.MinibatchData, ArrayMixin):
    '''
    Holds a minibatch of input data. This is never directly created, but
    only returned by :class:`MinibatchSource` instances.
    '''

    @property
    def num_sequences(self):
        '''
        The number of sequences in this minibatch
        '''
        return self.number_of_sequences

    @property
    def num_samples(self):
        '''
        The number of samples in this minibatch
        '''
        return self.number_of_samples

    @property
    def value(self):
        '''
        The value of the minibatch as a NumPy array.
        '''
        return value_to_seq(self.data)

    @property
    def shape(self):
        '''
        The shape of the data in this minibatch as tuple.
        '''
        return self.data.shape().dimensions()

    @property
    def mask(self):
        '''
        The mask object of the minibatch. In it, `2` marks the beginning of a
        sequence, `1` marks a sequence element as valid, and `0` marks it as
        invalid.
        '''
        return self.data.mask().to_ndarray()

    @property
    def end_of_sweep(self):
        '''
        Indicates whether the data in this minibatch comes from a sweep end
        or crosses a sweep boundary (and as a result includes data from
        different sweeps).
        '''
        return self.sweep_end

    @property
    def is_sparse(self):
        '''
        Whether the data in this minibatch is sparse.
        '''
        return self.data.is_sparse()

    def __len__(self):
        return self.num_sequences

class MinibatchSource(cntk_py.MinibatchSource):
    '''
    !DEPRECATED! The following multi-parameter constructor is deprecated,
    Please use :class:`~cntk.io.MinibatchSourceConfig` instead to create 
    an instance of :class:`~cntk.io.MinibatchSource`.

    MinibatchSource(deserializers=None, randomize=True, randomization_window=cntk.io.DEFAULT_RANDOMIZATION_WINDOW, 
        epoch_size=cntk.io.INFINITELY_REPEAT, distributed_after=cntk.io.INFINITE_SAMPLES, multithreaded_deserializer=None)

    Args:
        deserializers (`list`, defaults to empty): list of deserializers
        randomize (`bool`, defaults to `True`): randomize before every epoch
        randomization_window (int): size of window that reader will shuffle, ignored if `randomize`
          is `False`
        sample_based_randomization_window (`bool`c, defaults to `False`): specifies how to interpret
          `randomization_window`. If `True`, the size of the randomization window is interpreted as a certain
          number of samples, otherwise -- as a number of chunks. Similarly to `randomization_window`,
          this parameter is ignored, when `randomize` is `False`
        epoch_size (`int`, defaults to cntk.io.INFINITELY_REPEAT): maximum number of input samples (not 'label samples')
          the reader can produce. After the specified number of samples has been reached, the reader returns empty 
          minibatches on subsequent calls to :func:`~cntk.io.MinibatchSource.next_minibatch`. **Important:** 
          `see <https://github.com/Microsoft/CNTK/wiki/BrainScript-epochSize-and-Python-epoch_size-in-CNTK>`_ 
          for a description of input and label samples. 
        multithreaded_deserializer (`bool`, defaults to `False`): using multi threaded deserializer
        frame_mode (`bool`, defaults to `False`): Specifies if data should be randomized and returned at the frame
          or sequence level. When  true , input sequence are split into frames.
        truncation_length (`int`, , defaults to `0`): Specifies the truncation length in samples for BPTT 
          (positive integer). If greater than zero `frame_mode` cannot be used at the same time.
    '''
    def __init__(self, 
        deserializers=None,
        randomize=True,
        randomization_window=DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS,
        sample_based_randomization_window=False,
        epoch_size=INFINITELY_REPEAT,
        distributed_after=None,             #ignored, do not set this parameter
        multithreaded_deserializer=False,
        frame_mode=False,
        truncation_length=0):

        import warnings
        warnings.warn('This constuctor is deprecated and will be removed '
            'in future versions. To created a minibatch source, please use '
            'MinibatchSourceConfig instead', DeprecationWarning)

        config = MinibatchSourceConfig(randomize) \
                    .set_max_samples(epoch_size) \
                    .set_multithreaded(multithreaded_deserializer) \
                    .set_frame_mode(frame_mode);

        if truncation_length != 0:
            config.set_truncation_length(truncation_length);

        if (randomize and sample_based_randomization_window):
            config.set_randomization_window_in_samples(randomization_window);
        elif (randomize and not sample_based_randomization_window):
            config.set_randomization_window_in_chunks(randomization_window);

        if isinstance(deserializers, (list,tuple)):
            for deserializer in deserializers:
                config.add_deserializer(deserializer)
        elif deserializers is not None:
            config.add_deserializer(deserializers)

        self.source = cntk_py.create_composite_minibatch_source(config)
        self._streams = None


    def stream_infos(self):
        '''
        Describes the streams 'this' minibatch source produces.

        Returns:
            A `list` of instances of :class:`~cntk.cntk_py.StreamInformation`
        '''
        return self.source.stream_infos()

    @property
    def streams(self):
        '''
        Describes the streams 'this' minibatch source produces.

        Returns:
            A `dict` mapping input names to instances of 
            :class:`~cntk.cntk_py.StreamInformation`
        '''
        if self._streams is None:
            from ..utils import Record
            self._streams = Record(**dict((info.m_name, info) for info in  self.stream_infos()))

        return self._streams;

    def stream_info(self, name):
        '''
        Gets the description of the stream with given name.
        Throws an exception if there are none or multiple streams with this
        same name.
        '''
        return self.source.stream_info(name)

    def __getitem__(self, name):
        '''
        Return the :class:`~cntk.cntk_py.StreamInformation` for the given stream name

        Args:
            name (str): stream name to fetch :class:`~cntk.cntk_py.StreamInformation` for
        '''
        return self.stream_info(name)

    @typemap
    def next_minibatch(self, minibatch_size_in_samples,
            input_map=None, device=None, num_data_partitions=None, partition_index=None):
        '''
        Reads a minibatch that contains data for all input streams.  The
        minibatch size is specified in terms of #samples and/or #sequences for the
        primary input stream; value of 0 for #samples/#sequences means
        unspecified.  In case the size is specified in terms of both #sequences
        and #samples, the smaller of the 2 is taken.  An empty map is returned
        when the MinibatchSource has no more data to return.

        Args:
            minibatch_size_in_samples (int): number of samples to retrieve for
              the next minibatch. Must be > 0. **Important:** `click here <https://github.com/Microsoft/CNTK/wiki/BrainScript-minibatchSize-and-Python-minibatch_size_in_samples-in-CNTK>`_ for a full description of this parameter. 
            input_map (dict): mapping of :class:`~cntk.ops.variables.Variable`
              to :class:`~cntk.cntk_py.StreamInformation` which will be used to convert the
              returned data.
            device (`DeviceDescriptor`, defaults to `None`): CNTK DeviceDescriptor
            num_data_partitions: Used for distributed training, indicates into how many partitions
              the source should split the data.
            partition_index (`int`, defaults to `None`): Used for distributed training, indicates data from which partition to take.

        Returns:
            cntk.io.MinibatchData:
            A mapping of :class:`~cntk.cntk_py.StreamInformation` to :class:`MinibatchData` if
            `input_map` was not specified. Otherwise, the returned value will
            be a mapping of :class:`~cntk.ops.variables.Variable` to class:`MinibatchData`.
        '''
        if device is None:
            device = use_default_device()

        if num_data_partitions is None:
            num_data_partitions = 1

        if partition_index is None:
            partition_index = 0

        mb = self.source.get_next_minibatch(0,
                minibatch_size_in_samples, num_data_partitions, partition_index, device)

        if not mb:
            return mb;

        if not input_map:
            return mb

        return { key : mb[value] for (key, value) in input_map.items() }

    def get_checkpoint_state(self):
        '''
        Gets the checkpoint state of the MinibatchSource.

        Returns:
            cntk.cntk_py.Dictionary:
            A :class:`~cntk.cntk_py.Dictionary` that has the checkpoint state of the MinibatchSource
        '''
        return self.source.get_checkpoint_state()

    def restore_from_checkpoint(self, checkpoint):
        '''
        Restores the MinibatchSource state from the specified checkpoint.

        Args:
            checkpoint (:class:`~cntk_py.Dictionary`): checkpoint to restore from
        '''
        self.source.restore_from_checkpoint(checkpoint)

    @property
    def is_distributed(self):
        '''
        Whether the minibatch source is running distributed
        '''
        return self.source.is_distributed()

    @property
    def current_position(self):
        '''
        Gets current position in the minibatch source.

        Returns:
            Minibatch position :class:`~cntk.cntk_py.Dictionary` on the global timeline.
        '''
        return self.get_checkpoint_state()

    @current_position.setter
    def current_position(self, position):
        '''
        Sets current position in the minibatch source.

        Args:
            position (:class:`~cntk.cntk_py.Dictionary`): position returned from :func:`~get_current_position`.
        '''
        self.restore_from_checkpoint(position)


class _MinibatchSource(MinibatchSource):
    # this will eventually reaplace the multi-parameter __init__ 
    # method of the parent, after which this class can be removed.
    def __init__(self, config):
        self.source = cntk_py.create_composite_minibatch_source(config)
        self._streams = None


def _py_dict_to_cntk_dict(py_dict):
    '''
    Converts a Python dictionary into a CNTK Dictionary whose values are CNTK DictionaryValue instances.

    Args:
        py_dict (dict): a dictionary to be converted.

    Returns:
        cntk_py.Dictionary:
        A :class:`~cntk_py.Dictionary` that has been converted from the input `dict`
    '''
    res = cntk_py.Dictionary()
    for k, v in py_dict.items():
        if isinstance(v, dict):
            res[k] = cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(v))
        # TODO: add support to list of lists ?
        elif isinstance(v, list):
            res[k] = cntk_py.DictionaryValue([cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(e) if isinstance(e, dict) else e) for e in v])
        else:
            res[k] = cntk_py.DictionaryValue(v)
    return res

@unique
class TraceLevel(Enum):

    Error = cntk_py.MinibatchSourceConfig.Error
    Warning = cntk_py.MinibatchSourceConfig.Warning
    Info = cntk_py.MinibatchSourceConfig.Info
    

class MinibatchSourceConfig(cntk_py.MinibatchSourceConfig):
    '''
    A configuration that can be used to instantiate the CNTK built-in 
    composite minibatch source.

    Args:
        randomize (`bool`, defaults to `True`): enables/disables reader randomization,
          when randomize is `True` also sets the randomization window size to 
          `cntk.io.DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS`.
    '''
    def __init__(self, randomize=True):
        super(MinibatchSourceConfig, self).__init__(randomize)

    def set_max_samples(self, value):
        '''
        Sets the limit on the maximum number of input samples (not 'label samples') the reader 
        can produce (the default value is `cntk.io.INFINITELY_REPEAT`). After the minimum of 
        max samples and max sweeps has been reached, the reader returns empty minibatches on subsequent
        calls to :func:`~cntk.io.MinibatchSource.next_minibatch`. 

        **Important:** `see <https://github.com/Microsoft/CNTK/wiki/BrainScript-epochSize-and-Python-epoch_size-in-CNTK>`_ 
        for a description of input and label samples. 
        '''
        super(MinibatchSourceConfig, self).set_max_samples(value)
        return self

    def get_max_samples(self):
        '''
        Gets the limit on the maximum allowed number of input samples the reader can produce.
        '''
        return super(MinibatchSourceConfig, self).get_max_samples()

    def set_max_sweeps(self, value):
        '''
        Sets the limit on the maximum allowed number of sweeps over the input dataset 
        (the default value is `cntk.io.INFINITELY_REPEAT`). After the minimum of max samples 
        and max sweeps has been reached, reader returns empty minibatches on subsequent 
        calls to :func:`~cntk.io.MinibatchSource.next_minibatch`.
        '''
        super(MinibatchSourceConfig, self).set_max_sweeps(value)
        return self

    def get_max_sweeps(self):
        '''
        Gets the limit on the maximum allowed number of sweeps over the input dataset.
        '''
        return super(MinibatchSourceConfig, self).get_max_sweeps()

    def set_randomization_window_in_samples(self, value):
        '''
        Sets the size of the randomization window in samples and enables randomization (this 
        setter overrides the ``randomize`` value used to create this config). If the randomization 
        window was previously specified in chunks, this new setting takes precedence, effectively
        overwriting any previous value.
        '''
        super(MinibatchSourceConfig, self).set_randomization_window_in_samples(value)
        return self

    def set_randomization_window_in_chunks(self, value):
        '''
        Sets the size of the randomization window in chunks and enables randomization (this 
        setter overrides the ``randomize`` value used to create this config). If the randomization 
        window was previously specified in samples, this new setting takes precedence, effectively
        overwriting any previous value.
        '''
        super(MinibatchSourceConfig, self).set_randomization_window_in_chunks(value)
        return self

    def set_trace_level(self, value):
        '''
        Sets the output verbosity level (an instance of :class:`cntk.io.TraceLevel`).
        '''
        if isinstance(value, TraceLevel):
            super(MinibatchSourceConfig, self).set_trace_level(value.value)
        else:
            super(MinibatchSourceConfig, self).set_trace_level(value)
        return self

    def set_truncation_length(self, value):
        '''
        Enables the truncation and sets the truncation length in samples (only applicable for BPTT,
        cannot be used in frame mode).
        '''
        super(MinibatchSourceConfig, self).set_truncation_length(value)
        return self

    def set_frame_mode(self, value=True):
        '''
        Toggles the frame mode on and off. If the frame mode is enabled the input data will be processed
        as individual frames ignoring all sequence information (this option cannot be used for BPTT).
        '''
        super(MinibatchSourceConfig, self).set_frame_mode(value)
        return self

    def set_multithreaded(self, value=True):
        '''
        Specifies if the deserialization should be done on a single or multiple threads.
        '''
        super(MinibatchSourceConfig, self).set_multithreaded(value)
        return self

    def add_deserializer(self, value):
        '''
        Adds a deserializer to be used in the composite reader.
        '''
        super(MinibatchSourceConfig, self).add_deserializer(value)
        return self

    def create_minibatch_source(self):
        '''
        Creates an instance of the CNTK built-in composite minibatch source.

        Returns:
            An instance of :class:`cntk.io.MinibatchSource` built using
            parameters specified in this configuration.
        '''
        return _MinibatchSource(self)

def HTKFeatureDeserializer(streams):
    '''
    Configures the HTK feature reader that reads speech data from scp files.

    Args:
        streams: any dictionary-like object that contains a mapping from stream names
            to :class:`StreamDef` objects. Each StreamDef object configures a feature stream.
    '''
    feat = []
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None: raise ValueError("HTKFeatureDeserializer does not support steam names")
        if 'scp' not in stream: raise ValueError("No scp files specified for HTKFeatureDeserializer")
        dimension = stream.dim
        scp_file = stream['scp']
        broadcast = stream['broadcast'] if 'broadcast' in stream else False
        left_context, right_context = stream.context if 'context' in stream else (0, 0)
        feat.append(cntk_py.HTKFeatureConfiguration(stream_name, scp_file, dimension, left_context, right_context, broadcast))
    if len(feat) == 0:
        raise ValueError("no feature streams found")
    return cntk_py.htk_feature_deserializer(feat)

def HTKMLFDeserializer(label_mapping_file, streams):
    '''
    Configures an HTK label reader that reads speech HTK format MLF (Master Label File)

    Args:
        label_mapping_file (str): path to the label mapping file
        streams: any dictionary-like object that contains a mapping from stream names
            to :class:`StreamDef` objects. Each StreamDef object configures a label stream.
    '''
    if len(streams) != 1: raise ValueError("HTKMLFDeserializer only accepts a single stream")
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None: raise ValueError("HTKMLFDeserializer does not support steam names")
        dimension = stream.dim
        if 'mlf' not in stream: raise ValueError("No master label files specified for HTKMLFDeserializer")
        master_label_files = stream['mlf']
        if not isinstance(master_label_files,list):
            master_label_files = [master_label_files]
        return cntk_py.htk_mlf_deserializer(stream_name, label_mapping_file, dimension, master_label_files)


def ImageDeserializer(filename, streams):
    '''
    Configures the image reader that reads images and corresponding
    labels from a file of the form::

         <full path to image> <tab> <numerical label (0-based class id)>
    or::

        sequenceId <tab> path <tab> label

    Args:
        filename (str): file name of the map file that associates images to
         classes

    See also:
        `Image reader definition <https://github.com/microsoft/cntk/wiki/Image-reader>`_
    '''
    image_stream_name = None
    label_stream_name = '_ignore_labels_'
    num_labels = 2
    transforms = []
    for key in streams:
        s = streams[key]
        alias = s.stream_alias
        if alias == "image":
            image_stream_name = key
            transforms = s.transforms
        elif alias == "label":
            label_stream_name = key
            num_labels = s.dim
        else:
            raise ValueError("ImageDeserializer: invalid field name '{}', allowed are 'image' and 'label'".format(alias))
    if image_stream_name is None:
        raise ValueError("ImageDeserializer: stream name ('image' or 'label') must be specified")
    return cntk_py.image_deserializer(filename, label_stream_name, num_labels, image_stream_name, transforms)

def CTFDeserializer(filename, streams):
    '''
    Configures the CNTK text-format reader that reads text-based files with lines of the form::

        [Sequence_Id] (Sample)+

    where::

        Sample=|Input_Name (Value )*

    Args:
        filename (str): file name containing the text input

    See also:
        `CNTKTextReader format <https://github.com/microsoft/cntk/wiki/BrainScript-CNTKTextFormat-Reader>`_
    '''
    for k,s in streams.items():
        if s.stream_alias is None:
            raise ValueError("CTFDeserializer: stream name for key %s must be specified"%(k))
    sc = [cntk_py.StreamConfiguration(k, s.dim, s.is_sparse, s.stream_alias) for k,s in streams.items()]
    return cntk_py.ctf_deserializer(filename, sc)

# TODO: this should be a private class; use StreamDef instead
class StreamConfiguration(cntk_py.StreamConfiguration):
    '''
    Configuration of a stream in a text format reader.

    Args:
        name (str): name of this stream
        dim (int): dimensions of this stream. A text format reader reads data
          as flat arrays. If you need different shapes you can
          :func:`~cntk.ops.reshape` it later.
        is_sparse (bool, defaults to `False`): whether the provided data is sparse
          (`False` by default)
        stream_alias (str, defaults to ''): name of the stream in the file
    '''
    def __init__(self, name, dim, is_sparse=False, stream_alias=''):
        return super(StreamConfiguration, self).__init__(name, dim, is_sparse, stream_alias)

# stream definition for use in StreamDefs
# returns a record { stream_alias, is_sparse, optional shape, optional transforms, optional context, optional scp, optional mlf }
from cntk.utils import Record
def StreamDef(field=None, shape=None, is_sparse=False, transforms=None, context=None, scp=None, mlf=None, broadcast=None):
    '''
       Configuration of a stream for use with the builtin Deserializers.
       The meanings of some configuration keys have a mild dependency on the
       exact deserializer, and certain keys are meaningless for certain deserializers.

    Args:
        field (`str`, defaults to `None`): this is the name of the stream

         * for CTFDeserializer the name is inside the CTF file
         * for ImageDeserializer the acceptable names are `image` or `label`
         * for HTKFeatureDeserializer and HTKMLFDeserializer only the default
           value of None is acceptable

        shape (`int` or `tuple`, defaults to `None`): dimensions of this stream. HTKFeatureDeserializer,
         HTKMLFDeserializer, and CTFDeserializer read data
         as flat arrays. If you need different shapes you can
         :func:`~cntk.ops.reshape` it later.
        is_sparse (`bool`, defaults to `False`): whether the provided data is sparse.
         `False` by default, unless mlf is provided.
        transforms (`list`, defaults to `None`): list of transforms to be applied by the Deserializer.
         Currently only ImageDeserializer supports transforms.
        context (`tuple`, defaults to `None`): left and right context to consider when reading in HTK
         data. Only supported by HTKFeatureDeserializer.
        scp (`str` or `list`, defaults to `None`): scp files for HTK data
        mlf (`str` or `list`, defaults to `None`): mlf files for HTK data
        broadcast (`bool`, defaults to `None`): whether the features in this stream should be
         broadcast to the whole sequence (useful in e.g. ivectors with HTK)
    '''
    config = dict(stream_alias=field, is_sparse=is_sparse)
    if shape is not None:
        config['dim'] = shape
    if transforms is not None:
        config['transforms'] = transforms
    if context is not None:
        config['context'] = context
    if scp is not None:
        config['scp'] = scp
    if mlf is not None:
        config['mlf'] = mlf
        config['is_sparse'] = True
    if broadcast is not None:
        config['broadcast'] = broadcast
    return Record(**config)
    # TODO: we should always use 'shape' unless it is always rank-1 or a single rank's dimension
    # TODO: dim should be inferred from the file, at least for dense

# StreamDefs for use in constructing deserializers
# StreamDefs(query = StreamDef(...), labels = StreamDef(...), ...)
StreamDefs = Record

def _dense_to_str(data):
    return ' '.join(data.ravel(order='C').astype(np.str))


def _sparse_to_str(data):
    return ' '.join('%s:%s' % (k, v) for k, v in sorted(data.items()))


def _is_tensor(data):
    '''
    Checks whether the data is a tensor, i.e. whether it is a NumPy array or a
    list of NumPy arrays.

    Args:
        data: data to check

    Returns:
      bool:
      `True`, if it is a tensor.
    '''
    if isinstance(data, np.ndarray):
        return True

    if not isinstance(data, list):
        return False

    while len(data) > 0:
        # All but the innermost dimension's values have to be lists
        try:
            data[0][0]
        except:
            # We reached the innermost dimension
            try:
                data[0] + 0
                return True
            except:
                # Innermost type is not a number
                return False

        if isinstance(data, np.ndarray):
            return True

        if not isinstance(data[0], list):
            return False

        data = data[0]

    return True


def sequence_to_cntk_text_format(seq_idx, alias_tensor_map):
    '''
    Converts a list of NumPy arrays representing tensors of inputs into a
    format that is readable by :class:`~cntk.io.CTFDeserializer`.

    Args:
        seq_idx (int): number of current sequence
        alias_tensor_map (dict): maps alias (str) to tensor (ndarray). Tensors
          are assumed to have dynamic axis.

    Returns:
        str:
        String representation in `CNTKTextReader format <https://github.com/microsoft/cntk/wiki/BrainScript-CNTKTextFormat-Reader>`_
    '''

    max_seq_length = max(len(t) for t in alias_tensor_map.values())

    if max_seq_length == 0:
        return ''

    lines = []
    for elem_idx in range(0, max_seq_length):
        line = []

        for alias, tensor in sorted(alias_tensor_map.items()):
            if elem_idx >= len(tensor):
                # for this alias there no more sequence elements
                continue

            if _is_tensor(tensor):
                if not isinstance(tensor, np.ndarray):
                    tensor = np.asarray(tensor)
                to_str = _dense_to_str
            elif isinstance(tensor, list) and isinstance(tensor[0], dict):
                to_str = _sparse_to_str
            else:
                raise ValueError(
                    'expected a tensor (dense) or list of dicts (sparse), but got "%s"' % type(tensor))

            line.append('%s %s' % (alias, to_str(tensor[elem_idx])))

        lines.append('%i\t|' % seq_idx + ' |'.join(line))

    return '\n'.join(lines)
