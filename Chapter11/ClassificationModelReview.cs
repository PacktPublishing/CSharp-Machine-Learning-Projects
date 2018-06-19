using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClassificationModelReview
{
    class Program
    {
        static void Main(string[] args)
        {
            // sample input
            double[][] inputs =
            {
                new double[] { 0, 0 },
                new double[] { 1, 0 }, 
                new double[] { 0, 1 }, 
                new double[] { 1, 1 },
            };

            // sample binary output
            int[] outputs =
            { 
                0,
                1,
                1,
                0,
            };

            // sample binary output for Neural Network
            double[][] nnOutputs =
            {
                new double[] { 1, 0 },
                new double[] { 0, 1 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
            };

            // sample multinomial output
            int[] multiOutputs =
            {
                0,
                1,
                1,
                2,
            };

            // 1. Binary Logistic Regression
            var learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                MaxIterations = 100
            };
            var model = learner.Learn(inputs, outputs);

            var preds = model.Decide(inputs);
            Console.WriteLine("\n\n*Binary Logistic Regression Predictions: {0}", String.Join(", ", preds));

            // 2. Multinomial Logistic Regression
            var learner2 = new MultinomialLogisticLearning<GradientDescent>()
            {
                MiniBatchSize = 4
            };
            var model2 = learner2.Learn(inputs, multiOutputs);

            var preds2 = model2.Decide(inputs);
            Console.WriteLine("\n\n*Multinomial Logistic Regression Predictions: {0}", String.Join(", ", preds2));

            // 3. Binary Naive Bayes Classifier
            var learner3 = new NaiveBayesLearning<NormalDistribution>();
            var model3 = learner3.Learn(inputs, outputs);

            var preds3 = model2.Decide(inputs);
            Console.WriteLine("\n\n*Binary Naive Bayes Predictions: {0}", String.Join(", ", preds3));

            // 4. RandomForest
            var learner4 = new RandomForestLearning()
            {
                NumberOfTrees = 3,

                CoverageRatio = 0.9,

                SampleRatio = 0.9

            };
            var model4 = learner4.Learn(inputs, outputs);

            var preds4 = model4.Decide(inputs);
            Console.WriteLine("\n\n*Binary RandomForest Classifier Predictions: {0}", String.Join(", ", preds4));

            // 5. SVM
            var learner5 = new SequentialMinimalOptimization<Gaussian>();
            var model5 = learner.Learn(inputs, outputs);

            var preds5 = model5.Decide(inputs);
            Console.WriteLine("\n\n*Binary SVM Predictions: {0}", String.Join(", ", preds5));

            // 6. Neural Network
            var network = new ActivationNetwork(
                new BipolarSigmoidFunction(2),
                2,
                1,
                2
            );

            var teacher = new LevenbergMarquardtLearning(network);

            Console.WriteLine("\n-- Training Neural Network");
            int numEpoch = 3;
            double error = Double.PositiveInfinity;
            for (int i = 0; i < numEpoch; i++)
            {
                error = teacher.RunEpoch(inputs, nnOutputs);
                Console.WriteLine("* Epoch {0} - error: {1:0.0000}", i + 1, error);
            }

            double[][] nnPreds = inputs.Select(
                x => network.Compute(x)
            ).ToArray();

            int[] preds6 = nnPreds.Select(
                x => x.ToList().IndexOf(x.Max())
            ).ToArray();

            Console.WriteLine("\n\n*Binary Neural Network Predictions: {0}", String.Join(", ", preds6));


            Console.WriteLine("\n\n\n\nDONE!!");
            Console.ReadKey();
        }
    }
}
