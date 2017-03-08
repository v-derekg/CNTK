//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "PerformanceProfiler.h"

namespace CNTK
{
    Evaluator::Evaluator(const FunctionPtr& model, const FunctionPtr& evaluationFunction,
        const std::vector<ProgressWriterPtr>& progressWriters)
        : m_model(model),
        m_evaluationFunction(evaluationFunction),
        m_aggregatedTestEvalCriterionValue(),
        m_progressWriters(progressWriters.begin(), progressWriters.end())
    {
        // By default we set the number of threads to hardware concurrency.
        if (!Internal::MaxNumCPUThreadsSet())
            SetMaxNumCPUThreads(std::thread::hardware_concurrency());

        std::vector<Variable> combinedFunctionArgs;
        if (m_model) // model is optional, since it may not be adding any information on top of lossFunction
            combinedFunctionArgs = m_model->Outputs();

        if (m_evaluationFunction)
        {
            combinedFunctionArgs.push_back(m_evaluationFunction);

            if (!m_evaluationFunction->Output().DynamicAxes().empty())
            {
                m_aggregatedEvaluationFunction = ReduceSum(m_evaluationFunction);
                combinedFunctionArgs.push_back(m_aggregatedEvaluationFunction);
                m_testSampleCountVar = m_evaluationFunction;
            }
            else
            {
                m_aggregatedEvaluationFunction = m_evaluationFunction;
                m_testSampleCountVar = m_evaluationFunction->RootFunction()->Inputs()[0];
                if (model->Output() != m_testSampleCountVar)
                    combinedFunctionArgs.push_back(m_testSampleCountVar);
            }

            m_aggregatedTestEvalCriterionValue = std::make_shared<Accumulator>();
        }

        m_combinedEvalFunction = Combine(combinedFunctionArgs);
    }

    static size_t GetSampleCount(const Variable& var, const ValuePtr& value)
    {
        auto valueDataShape = value->Shape();
        size_t numMaskedSamples = value->MaskedCount();
        size_t numSamplesInDataArrayView = valueDataShape.SubShape(var.Shape().Rank()).TotalSize();
        if (numMaskedSamples > numSamplesInDataArrayView)
            LogicError("Number (%d) of masked values cannot exceed the number (%d) of samples that the Value object's Data NDArrayView can hold.",
            (int)numMaskedSamples, (int)numSamplesInDataArrayView);

        return (numSamplesInDataArrayView - numMaskedSamples);
    }

    static std::unordered_map<Variable, ValuePtr> GetInputs(const std::unordered_map<Variable, MinibatchData>& arguments)
    {
        std::unordered_map<Variable, ValuePtr> inputs(arguments.size());
        for (const auto& kv : arguments)
        {
            inputs[kv.first] = kv.second.data;
        }
        return inputs;
    }

    double Evaluator::TestMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return TestMinibatch(GetInputs(arguments), computeDevice);
    }

    double Evaluator::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        size_t sampleCount = 0;
        double error = TestMinibatch(arguments, computeDevice, sampleCount);
        return error / sampleCount;
    }

    double Evaluator::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice, size_t& sampleCount, bool distributed)
    {
        if (!distributed)
            return TestLocalMinibatch(arguments, computeDevice, sampleCount);
        else
        {
            std::array<double, 2> errorAndCount;
            size_t localSampleCount = 0;

            errorAndCount[0] = TestLocalMinibatch(arguments, computeDevice, localSampleCount);
            errorAndCount[1] = static_cast<double>(localSampleCount);

            auto value = std::vector<NDArrayViewPtr>{ MakeSharedObject<NDArrayView>(NDShape{ 2 }, errorAndCount.data(), 2, DeviceDescriptor::CPUDevice()) };

            DistributedCommunicatorPtr communicator = MPICommunicator();
            communicator->Barrier();
            communicator->AggregateInPlace(value, communicator->Workers());
            communicator->Barrier();

            sampleCount = static_cast<size_t>(errorAndCount[1]);
            return errorAndCount[0];
        }
    }

    double Evaluator::TestLocalMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice, size_t& sampleCount)
    {
        if (!m_aggregatedEvaluationFunction)
            InvalidArgument("Trainer::TestMinibatch: Cannot test when no evaluation function was specified during 'this' trainer's construction.");

        // TODO: Should we refactor this code that is somewhat similar to the prologue of the TrainMinibatch function
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedEvaluationFunction, nullptr }, { m_testSampleCountVar, nullptr } };

        m_combinedEvalFunction->Forward(arguments, outputs, computeDevice);
        const ValuePtr& aggregateEvalCriterionValue = outputs[m_aggregatedEvaluationFunction];
        sampleCount = GetSampleCount(m_testSampleCountVar, outputs[m_testSampleCountVar]);

        UpdateTestProgress(sampleCount, aggregateEvalCriterionValue, computeDevice);

        // TODO: it is not optimal to return average evaluation after each minibatch, since it potentially requires a
        // roundtrip to GPU. A better approach would be to have a separate method to return the average evaluation on
        // demand, as done for training. However, removing the below return is an API breaking change.
        return aggregateEvalCriterionValue->AsScalar<double>();
    }

    void Evaluator::UpdateTestProgress(size_t numSamples, const ValuePtr& evalCriterion, const DeviceDescriptor& computeDevice)
    {
        if (numSamples == 0)
        {
            return;
        }

        if (m_aggregatedTestEvalCriterionValue)
        {
            m_aggregatedTestEvalCriterionValue->Update(evalCriterion, computeDevice);
        }

        for (auto& progressWriter : m_progressWriters)
        {
            progressWriter->UpdateTest(numSamples, m_aggregatedTestEvalCriterionValue);
        }
    }

    void Evaluator::SummarizeTestProgress()
    {
        for (auto& progressWriter : m_progressWriters)
        {
            progressWriter->WriteTestSummary(m_aggregatedTestEvalCriterionValue);
        }

        if (m_aggregatedTestEvalCriterionValue)
        {
            m_aggregatedTestEvalCriterionValue->Reset();
        }
    }

    EvaluatorPtr CreateEvaluator(const FunctionPtr& model, const FunctionPtr& evaluationFunction,  const std::vector<ProgressWriterPtr>& progressWriters)
    {
        return MakeSharedObject<Evaluator>(model, evaluationFunction, progressWriters);
    }
}
