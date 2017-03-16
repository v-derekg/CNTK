//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Config.h"
#include "MinibatchSource.h"
#include "HeapMemoryProvider.h"
#include "ReaderShim.h"
#include "ReaderConstants.h"
#include <tuple>
#include "Value.h"
#include "MPIWrapper.h"
#include "PerformanceProfiler.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    const size_t MinibatchSource::DefaultRandomizationWindowInChunks = g_4GB / g_32MB;
    const size_t  MinibatchSource::InfinitelyRepeat = g_infinity;
    const size_t  MinibatchSource::FullDataSweep = g_dataSweep;


    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(0, minibatchSizeInSamples, device);
    }

    const std::unordered_map<StreamInformation, MinibatchData>& MinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences, size_t minibatchSizeInSamples, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return GetNextMinibatch(minibatchSizeInSequences, minibatchSizeInSamples, 1, 0, device);
    }

    MinibatchSourceConfig::MinibatchSourceConfig(bool randomize/* = true*/)
    {
        if (randomize)
            SetRandomizationWindowInChunks(MinibatchSource::DefaultRandomizationWindowInChunks);
        else 
            m_dict[L"randomize"] = false;
    }


    MinibatchSourceConfig& MinibatchSourceConfig::SetMaxSamples(size_t value)
    {
        m_dict[L"maxSamples"] = value;
        return *this;
    }

    size_t MinibatchSourceConfig::GetMaxSamples() const
    {
        if (m_dict.Contains(L"maxSamples")) 
            return m_dict[L"maxSamples"].Value<size_t>();

        return MinibatchSource::InfinitelyRepeat;
    }

    MinibatchSourceConfig&  MinibatchSourceConfig::SetMaxSweeps(size_t value)
    {
        m_dict[L"maxSweeps"] = value;
        return *this;
    }

    size_t MinibatchSourceConfig::GetMaxSweeps() const
    {
        if (m_dict.Contains(L"maxSweeps"))
            return m_dict[L"maxSweeps"].Value<size_t>();

        return MinibatchSource::InfinitelyRepeat;
    }

    MinibatchSourceConfig&  MinibatchSourceConfig::SetRandomizationWindowInSamples(size_t value)
    {
        m_dict[L"randomize"] = true;
        m_dict[L"randomizationWindow"] = value;
        m_dict[L"sampleBasedRandomizationWindow"] = true;
        return *this;
    }

    MinibatchSourceConfig&  MinibatchSourceConfig::SetRandomizationWindowInChunks(size_t value)
    {
        m_dict[L"randomize"] = true;
        m_dict[L"randomizationWindow"] = value;
        m_dict[L"sampleBasedRandomizationWindow"] = false;
        return *this;
    }

    MinibatchSourceConfig& MinibatchSourceConfig::SetTruncationLength(size_t value) 
    {
        if (m_dict.Contains(L"frameMode") && m_dict[L"frameMode"].Value<bool>())
            LogicError("MinibatchSourceConfig: truncation and frame mode are mutually exclusive options.");

        m_dict[L"truncated"] = true;
        m_dict[L"truncationLength"] = value;
        return *this;
    }

    MinibatchSourceConfig&  MinibatchSourceConfig::SetTraceLevel(TraceLevel value)
    {
        m_dict[L"traceLevel"] = size_t(value);
        return *this;
    }

    CNTK_API MinibatchSourceConfig& MinibatchSourceConfig::SetFrameMode(bool value/* = true*/)
    {
        if (m_dict.Contains(L"truncated") && m_dict[L"truncated"].Value<bool>())
            LogicError("MinibatchSourceConfig: truncation and frame mode are mutually exclusive options.");

        m_dict[L"frameMode"] = value;
        return *this;
    }

    CNTK_API MinibatchSourceConfig& MinibatchSourceConfig::SetMultithreaded(bool value/* = true*/)
    {
        m_dict[L"multiThreadedDeserialization"] = value;
        return *this;
    }

    MinibatchSourceConfig&  MinibatchSourceConfig::AddDeserializer(const Deserializer& deserializer)
    {

        if (!m_dict.Contains(L"deserializers"))
            m_dict[L"deserializers"] = std::vector<DictionaryValue>();

        m_dict[L"deserializers"].Value<std::vector<DictionaryValue>>().push_back(deserializer);
        return *this;
    }

    const StreamInformation& MinibatchSource::StreamInfo(const std::wstring& streamName)
    {
        std::unordered_set<const StreamInformation*> matchingStreamInfos;
        auto& allStreamInfos = StreamInfos();
        for (auto& streamInfo : allStreamInfos)
        {
            if (streamInfo.m_name == streamName)
                matchingStreamInfos.insert(&streamInfo);
        }

        if (matchingStreamInfos.empty())
            RuntimeError("No stream found matching given name '%S'.", streamName.c_str());

        if (matchingStreamInfos.size() > 1)
            RuntimeError("Multiple streams (%d) found matching given name '%S'.", (int)matchingStreamInfos.size(), streamName.c_str());

        return *(*(matchingStreamInfos.begin()));
    }

    const StreamInformation& MinibatchSource::StreamInfo(const Variable& variableToMatch)
    {
        std::unordered_set<const StreamInformation*> matchingStreamInfos;
        auto& allStreamInfos = StreamInfos();
        for (auto& streamInfo : allStreamInfos)
        {
            bool streamHasSparseData = (streamInfo.m_storageFormat != StorageFormat::Dense);
            if ((streamInfo.m_elementType == variableToMatch.GetDataType()) && (streamInfo.m_sampleLayout == variableToMatch.Shape()) && (streamHasSparseData == variableToMatch.IsSparse()))
                matchingStreamInfos.insert(&streamInfo);
        }

        if (matchingStreamInfos.empty())
            RuntimeError("No stream found matching given Variable '%S'.", variableToMatch.AsString().c_str());

        if (matchingStreamInfos.size() > 1)
            RuntimeError("Multiple streams (%d) found matching given Variable '%S'.", (int)matchingStreamInfos.size(), variableToMatch.AsString().c_str());

        return *(*(matchingStreamInfos.begin()));
    }

    MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration)
    {
        return MinibatchSourcePtr(new CompositeMinibatchSource(configuration));
    }

    MinibatchSourcePtr CreateCompositeMinibatchSource(const MinibatchSourceConfig& configuration)
    {
        return MinibatchSourcePtr(new CompositeMinibatchSource(configuration));
    }

    /*static*/ const std::wstring CompositeMinibatchSource::PositionAttributeName = L"minibatchSourcePosition";

    CompositeMinibatchSource::CompositeMinibatchSource(const Dictionary& configuration)
        : m_epochEndReached(false),
          m_prevMinibatchSize(0),
          m_maxNumSamplesToRead(MinibatchSource::InfinitelyRepeat),
          m_maxNumSweepsToRead(MinibatchSource::InfinitelyRepeat),
          m_truncationLength(0),
          m_numWorkers(1),
          m_workerRank(0),
          m_restorePosition(0)
    {
        // The CNTK reader implementation requires for each deserializer both the module and deserializer type be specified
        // This is redundant and the V2 API users will just specify type from which the module is automatically inferred
        // TODO: This should be done in the same manner for CNTK exe as well.
        Dictionary augmentedConfiguration = configuration;
        auto& deserializerConfigurations = augmentedConfiguration[L"deserializers"].Value<std::vector<DictionaryValue>>();
        for (auto& deserializerConfig : deserializerConfigurations)
        {
            static const std::unordered_map<std::wstring, std::wstring> deserializerTypeNameToModuleNameMap = {
                { L"CNTKTextFormatDeserializer", L"CNTKTextFormatReader" },
                { L"ImageDeserializer",          L"ImageReader"          },
                { L"HTKFeatureDeserializer",     L"HTKDeserializers"     },
                { L"HTKMLFDeserializer",         L"HTKDeserializers"     },
            };

            auto& deserializerConfigDict = deserializerConfig.Value<Dictionary>();
            auto deserializerTypeName = deserializerConfigDict[L"type"].Value<std::wstring>();
            if (deserializerTypeName == L"ImageDeserializer")
            {
                // Add a transpose transform since the image data in read in HWC (CWH in column major format) form while 
                // the CNTK convolution engive supports WHC (in column-major format)
                auto& inputStreamsConfig = deserializerConfigDict[L"input"].Value<Dictionary>();
                auto& streamsMap = *(inputStreamsConfig.m_dictionaryData);
                for (auto& inputStreamEntry : streamsMap)
                {
                    auto& inputStreamConfig = inputStreamEntry.second.Value<Dictionary>();
                    if (inputStreamConfig.Contains(L"transforms"))
                    {
                        auto& transforms = inputStreamConfig[L"transforms"].Value<std::vector<DictionaryValue>>();

                        // Add the transpose transform
                        Dictionary transposeTransform;
                        transposeTransform[L"type"] = L"Transpose";
                        transforms.push_back(transposeTransform);
                    }
                }

            }

            if (deserializerTypeNameToModuleNameMap.find(deserializerTypeName) == deserializerTypeNameToModuleNameMap.end())
                InvalidArgument("Unknown deserializer type '%S' specified for CNTK built-in composite MinibatchSource construction.", deserializerTypeName.c_str());

            deserializerConfigDict[L"module"] = deserializerTypeNameToModuleNameMap.at(deserializerTypeName);
        }

        ConfigParameters config;
        std::wstringstream s;
        for (const auto& keyValuePair : *(augmentedConfiguration.m_dictionaryData))
            AddConfigString(s, keyValuePair.first, keyValuePair.second, 0);

        config.Parse(msra::strfun::utf8(s.str()));

        const wchar_t* epochSizeConfigurationKey = L"epochSize";
        if (augmentedConfiguration.Contains(epochSizeConfigurationKey))
            m_maxNumSamplesToRead = augmentedConfiguration[epochSizeConfigurationKey].Value<size_t>();

        const wchar_t* truncatedConfigurationKey = L"truncated";
        const wchar_t* truncationLengthConfigurationKey = L"truncationLength";
        if (augmentedConfiguration.Contains(truncatedConfigurationKey) &&
            augmentedConfiguration[truncatedConfigurationKey].Value<bool>() &&
            augmentedConfiguration.Contains(truncationLengthConfigurationKey))
        {
            m_truncationLength = augmentedConfiguration[truncationLengthConfigurationKey].Value<size_t>();
        }

        typedef Reader*(*CreateCompositeDataReaderProc)(const ConfigParameters* parameters);
        CreateCompositeDataReaderProc createReaderProc = (CreateCompositeDataReaderProc)Plugin().Load(L"CompositeDataReader", "CreateCompositeDataReader");
        std::shared_ptr<Microsoft::MSR::CNTK::Reader> compositeDataReader(createReaderProc(&config));

        m_compositeDataReaderStreamDescs = compositeDataReader->GetStreamDescriptions();
        for (auto streamDesc : m_compositeDataReaderStreamDescs)
            m_streamInfos.insert({ streamDesc->m_name, streamDesc->m_id, AsStorageFormat(streamDesc->m_storageType), AsDataType(streamDesc->m_elementType), AsNDShape(*(streamDesc->m_sampleLayout)) });

        m_shim = std::shared_ptr<ReaderShim<float>>(new ReaderShim<float>(compositeDataReader), [](ReaderShim<float>* x) { x->Destroy(); });
        m_shim->Init(config);
    }

    CompositeMinibatchSource::CompositeMinibatchSource(const MinibatchSourceConfig& configuration)
        : CompositeMinibatchSource(configuration.AsDictionary())
    {
        m_maxNumSamplesToRead = configuration.GetMaxSamples();
        m_maxNumSweepsToRead = configuration.GetMaxSweeps();
    }

    /*virtual*/ const std::unordered_map<StreamInformation, MinibatchData>&
    CompositeMinibatchSource::GetNextMinibatch(size_t minibatchSizeInSequences,
                                               size_t minibatchSizeInSamples,
                                               size_t numberOfWorkers,
                                               size_t workerRank,
                                               const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/) /*override*/
    {
        auto profGetMinibatch = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainGetMinibatch);

        m_minibatchData.clear();

        if (!m_epochEndReached)
        {
            if (minibatchSizeInSequences != 0)
                LogicError("GetNextMinibatch: Specifying minibatch size in #sequences is currently unsupported");

            if (minibatchSizeInSamples == 0)
                InvalidArgument("GetNextMinibatch: Requested minibatch size must be > 0.");

            if (m_prevMinibatchSize == 0)
            {
                EpochConfiguration epochConfig;
                epochConfig.m_numberOfWorkers = numberOfWorkers;
                epochConfig.m_workerRank = workerRank;
                epochConfig.m_minibatchSizeInSamples = minibatchSizeInSamples;
                epochConfig.m_truncationSize = m_truncationLength;
                epochConfig.m_allowMinibatchesToCrossSweepBoundaries = true;

                if (m_maxNumSamplesToRead == MinibatchSource::FullDataSweep)
                {
                    epochConfig.m_totalEpochSizeInSamples = Microsoft::MSR::CNTK::requestDataSize;
                }
                else if (m_maxNumSamplesToRead == MinibatchSource::InfinitelyRepeat)
                {
                    // Setting big value, but not the max in order to aviod bit overflow.
                    epochConfig.m_totalEpochSizeInSamples = std::numeric_limits<size_t>::max() / 2;
                }
                else 
                {
                    epochConfig.m_totalEpochSizeInSamples = m_maxNumSamplesToRead;
                }

                epochConfig.m_totalEpochSizeInSweeps = m_maxNumSweepsToRead;

                epochConfig.m_epochIndex = 0;

                m_matrices.clear();

                std::unordered_set<InputStreamDescription> inputs;
                for (const auto& s : m_streamInfos)
                {
                    auto inputStreamDescription = GetInputStreamDescription(s, device);
                    inputs.insert(inputStreamDescription);

                    if (s.m_elementType == DataType::Float)
                    {
                        auto iter = std::find_if(m_compositeDataReaderStreamDescs.begin(), m_compositeDataReaderStreamDescs.end(), [s](StreamDescriptionPtr& streamInfo) {
                            return streamInfo->m_id == s.m_id;
                        });
                        assert(iter != m_compositeDataReaderStreamDescs.end());

                        m_matrices.AddInput(
                            s.m_name,
                            std::make_shared<Matrix<float>>(0, 0, inputStreamDescription.GetDeviceId(), inputStreamDescription.GetMatrixType(), inputStreamDescription.GetMatrixFormat()),
                            std::make_shared<MBLayout>(),
                            *(*iter)->m_sampleLayout);
                    }
                    else
                        LogicError("GetNextMinibatch: Input of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
                }

                m_shim->StartEpoch(epochConfig, inputs);

                m_prevMinibatchSize = minibatchSizeInSamples;
                m_workerRank = workerRank;
                m_numWorkers = numberOfWorkers;
            }

            if (minibatchSizeInSamples != m_prevMinibatchSize || m_workerRank != workerRank || m_numWorkers != numberOfWorkers || m_restorePosition != 0)
            {
                std::map<std::wstring, int> inputDescriptions;
                for (const auto& s : m_streamInfos)
                    inputDescriptions[s.m_name] = AsCNTKImplDeviceId(device);

                ReaderConfiguration newConfig;
                newConfig.m_numberOfWorkers = numberOfWorkers;
                newConfig.m_workerRank = workerRank;
                newConfig.m_minibatchSizeInSamples = minibatchSizeInSamples;
                newConfig.m_truncationSize = m_truncationLength;
                newConfig.m_allowMinibatchesToCrossSweepBoundaries = true;

                if (m_restorePosition != 0)
                {
                    m_shim->SetCurrentSamplePosition(m_restorePosition);
                    m_restorePosition = 0;
                }

                m_shim->SetConfiguration(newConfig, inputDescriptions);

                m_prevMinibatchSize = minibatchSizeInSamples;
                m_workerRank = workerRank;
                m_numWorkers = numberOfWorkers;
            }

            auto hasData = m_shim->GetMinibatch(m_matrices);
            m_epochEndReached = m_shim->IsEndOfEpoch();

            if (m_epochEndReached && !hasData)
                return m_minibatchData;

            bool hasReachedSweepEnd = m_shim->IsEndOfSweep();

            for (const auto& s: m_streamInfos)
            {
                auto input = m_matrices.GetInput(s.m_name);
                auto& currentStreamInfo = s;

                ValuePtr minibatchValuePtr;
                if (!hasData)
                {
                    m_minibatchData[currentStreamInfo] = { nullptr, 0, 0 };
                    continue;
                }

                if (s.m_elementType == DataType::Float)
                {
                    auto matrix = dynamic_pointer_cast<Matrix<float>>(input.matrix);
                    if (!matrix)
                        LogicError("GetNextMinibatch: Invalid matrix type.");

                    minibatchValuePtr = MakeSharedObject<PackedValue>(s.m_sampleLayout, matrix, input.pMBLayout, /*readOnly =*/ false);

                    size_t numSamples = input.pMBLayout->GetActualNumSamples();
                    size_t numSequences = input.pMBLayout->GetNumSequences();

                    m_minibatchData[currentStreamInfo] = { minibatchValuePtr, numSequences, numSamples, hasReachedSweepEnd };
                }
                else
                    LogicError("GetNextMinibatch: Input of type other than DataType::Float is currently unsupported by the CNTK built-in composite MinibatchSource!");
            }
        }

        return m_minibatchData;
    }

    /*virtual*/ Dictionary CompositeMinibatchSource::GetCheckpointState() const /*override*/
    {
        Dictionary checkpointState;
        checkpointState[PositionAttributeName] = m_shim->GetCurrentSamplePosition();
        return checkpointState;
    }

    /*virtual*/ void CompositeMinibatchSource::RestoreFromCheckpoint(const Dictionary& checkpoint) /*override*/
    {
        auto checkpointedMinibatchSourcePosition = checkpoint[PositionAttributeName].Value<size_t>();
        m_shim->SetCurrentSamplePosition(checkpointedMinibatchSourcePosition);

        // Need to reinitialize, we also have to remember the current position because StartEpoch
        // effectively resets it.
        // TODO: Remove call to StartEpoch - this API is legacy.
        m_restorePosition = checkpointedMinibatchSourcePosition;
        m_epochEndReached = false;
        m_prevMinibatchSize = 0;
    }

    /* static */ ImageTransform ReaderCrop(const wchar_t* cropType,
            int cropSize, float sideRatio, float areaRatio,
            float aspectRatio, const wchar_t* jitterType)
    {
        ImageTransform crop;
        crop.Add(L"type", L"Crop",
            L"cropType", cropType,
            L"cropSize", cropSize,
            L"sideRatio", sideRatio,
            L"areaRatio", areaRatio,
            L"aspectRatio", aspectRatio,
            L"jitterType", jitterType);
        return crop;
    }

    /* static */ ImageTransform ReaderScale(int width,
            int height, int channels, const wchar_t* interpolations,
            const wchar_t* scaleMode, int padValue)
    {
        ImageTransform scale;
        scale.Add(L"type", L"Scale",
            L"width", width,
            L"height", height,
            L"channels", channels,
            L"interpolations", interpolations,
            L"scaleMode", scaleMode,
            L"padValue", padValue);
        return scale;
    }

    /* static */ ImageTransform ReaderMean(const wchar_t* meanFile)
    {
        ImageTransform mean;
        mean.Add(L"type", L"Mean", L"meanFile", meanFile);
        return mean;
    }

    /* static */ ImageTransform ReaderColor(float brightnessRadius,
            float contrastRadius, float saturationRadius)
    {
        ImageTransform color;
        color.Add(L"type", L"Color",
            L"brightnessRadius", brightnessRadius,
            L"contrastRadius", contrastRadius,
            L"saturationRadius", saturationRadius);
        return color;
    }

    Deserializer ImageDeserializer(const std::wstring& fileName, const std::wstring& labelStreamName, size_t numLabels, const std::wstring& imageStreamName, const std::vector<ImageTransform>& transforms)
    {
        Deserializer img;
        std::vector<DictionaryValue> actualTransforms;
        std::transform(transforms.begin(), transforms.end(), std::back_inserter(actualTransforms), [](ImageTransform t) { return static_cast<DictionaryValue>(t); });
        Dictionary labeldim;
        labeldim[L"labelDim"] = numLabels;
        Dictionary xforms;
        xforms[L"transforms"] = actualTransforms;
        Dictionary input;
        input.Add(imageStreamName.c_str(), xforms, labelStreamName.c_str(), labeldim);
        img.Add(L"type", L"ImageDeserializer", L"file", fileName, L"input", input);
        return img;
    }

    Deserializer CTFDeserializer(const std::wstring& fileName, const std::vector<StreamConfiguration>& streams)
    {
        Deserializer ctf;
        Dictionary input;
        for (const auto& s : streams)
        {
            const auto& key = s.m_streamName;
            Dictionary stream;
            stream[L"dim"] = s.m_dim;
            stream[L"format"] = s.m_isSparse ? L"sparse" : L"dense";
            if (!s.m_streamAlias.empty())
                stream[L"alias"] = s.m_streamAlias;
            input[key] = stream;
        }
        ctf.Add(L"type", L"CNTKTextFormatDeserializer", L"file", fileName, L"input", input);
        return ctf;
    }

    Deserializer HTKFeatureDeserializer(const std::vector<HTKFeatureConfiguration>& streams)
    {
        Deserializer htk;
        Dictionary input;
        for (const auto& s : streams)
        {
            const auto& key = s.m_streamName;
            Dictionary stream;
            std::vector<DictionaryValue> ctxWindow = { DictionaryValue(s.m_left), DictionaryValue(s.m_right) };
            stream.Add(L"scpFile", s.m_scp, L"dim", s.m_dim, L"contextWindow", ctxWindow, L"expandToUtterance", s.m_broadcast);
            input[key] = stream;
        }
        htk.Add(L"type", L"HTKFeatureDeserializer", L"input", input);
        return htk;
    }

    Deserializer HTKMLFDeserializer(const std::wstring& streamName, const std::wstring& labelMappingFile, size_t dimension, const std::vector<std::wstring>& mlfFiles)
    {
        Deserializer htk;
        Dictionary stream;
        Dictionary labels;
        labels.Add(L"labelMappingFile", labelMappingFile, L"dim", dimension);
        std::vector<DictionaryValue> actualFiles;
        std::transform(mlfFiles.begin(), mlfFiles.end(), std::back_inserter(actualFiles), [](const std::wstring& s) {return static_cast<DictionaryValue>(s); });
        if (actualFiles.size() > 1)
            labels[L"mlfFileList"] = actualFiles;
        else if (actualFiles.size() == 1)
            labels[L"mlfFile"] = actualFiles[0];
        else
            LogicError("HTKMLFDeserializer: No mlf files were specified");
        stream[streamName] = labels;
        htk.Add(L"type", L"HTKMLFDeserializer", L"input", stream);
        return htk;
    }
}
