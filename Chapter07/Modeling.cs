using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.Performance;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Distributions.Univariate;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;

namespace Modeling
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 60);

            // Read in the Audio Features dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.7\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "sample.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var featuresDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            Console.WriteLine("* Shape: {0}, {1}\n\n", featuresDF.RowCount, featuresDF.ColumnCount);

            string[] featureColumns = featuresDF.ColumnKeys.Where(x => !x.Equals("track_id") && !x.Equals("genre_top")).ToArray();
            IDictionary<string, int> targetVarCodes = new Dictionary<string, int>
            {
                { "Electronic", 0 },
                { "Experimental", 1 },
                { "Folk", 2 },
                { "Hip-Hop", 3 },
                { "Instrumental", 4 },
                { "International", 5 },
                { "Pop", 6 },
                { "Rock", 7 }
            };
            featuresDF.AddColumn("target", featuresDF.GetColumn<string>("genre_top").Select(x => targetVarCodes[x.Value]));

            // Create input and output variables from data frames, so that we can use them for Accord.NET MachineLearning models
            double[][] input = featuresDF.Columns[featureColumns].Rows.Select(
                x => Array.ConvertAll<object, double>(x.Value.ValuesAll.ToArray(), o => Convert.ToDouble(o))
            ).ValuesAll.ToArray();
            int[] output = featuresDF.GetColumn<int>("target").Values.ToArray();

            Accord.Math.Random.Generator.Seed = 0;

            // 1. Train a LogisticRegression Classifier
            Console.WriteLine("\n---- Logistic Regression Classifier ----\n");
            var logitSplitSet = new SplitSetValidation<MultinomialLogisticRegression, double[]>()
            {
                Learner = (s) => new MultinomialLogisticLearning<GradientDescent>()
                {
                    MiniBatchSize = 500
                },

                Loss = (expected, actual, p) => new ZeroOneLoss(expected).Loss(actual),

                Stratify = false,

                TrainingSetProportion = 0.8,

                ValidationSetProportion = 0.2,

            };

            var logitResult = logitSplitSet.Learn(input, output);

            var logitTrainedModel = logitResult.Model;

            // Store train & test set indexes to train other classifiers on the same train set
            // and test on the same validation set
            int[] trainSetIDX = logitSplitSet.IndicesTrainingSet;
            int[] testSetIDX = logitSplitSet.IndicesValidationSet;

            // Get in-sample & out-of-sample predictions and prediction probabilities for each class
            double[][] trainProbabilities = new double[trainSetIDX.Length][];
            int[] logitTrainPreds = new int[trainSetIDX.Length];
            for (int i = 0; i < trainSetIDX.Length; i++)
            {
                logitTrainPreds[i] = logitTrainedModel.Decide(input[trainSetIDX[i]]);
                trainProbabilities[i] = logitTrainedModel.Probabilities(input[trainSetIDX[i]]);
            }

            double[][] testProbabilities = new double[testSetIDX.Length][];
            int[] logitTestPreds = new int[testSetIDX.Length];
            for (int i = 0; i < testSetIDX.Length; i++)
            {
                logitTestPreds[i] = logitTrainedModel.Decide(input[testSetIDX[i]]);
                testProbabilities[i] = logitTrainedModel.Probabilities(input[testSetIDX[i]]);
            }

            Console.WriteLine(String.Format("train accuracy: {0:0.0000}", 1-logitResult.Training.Value));
            Console.WriteLine(String.Format("validation accuracy: {0:0.0000}", 1-logitResult.Validation.Value));

            // Build confusion matrix
            string[] confMatrix = BuildConfusionMatrix(
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray(), logitTestPreds, 8
            );

            System.IO.File.WriteAllLines(Path.Combine(dataDirPath, "logit-conf-matrix.csv"), confMatrix);

            // Calculate evaluation metrics
            int[][] logitTrainPredRanks = GetPredictionRanks(trainProbabilities);
            int[][] logitTestPredRanks = GetPredictionRanks(testProbabilities);

            double logitTrainMRRScore = ComputeMeanReciprocalRank(
                logitTrainPredRanks,
                output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()
            );
            double logitTestMRRScore = ComputeMeanReciprocalRank(
                logitTestPredRanks,
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray()
            );

            Console.WriteLine("\n---- Logistic Regression Classifier ----\n");
            Console.WriteLine(String.Format("train MRR score: {0:0.0000}", logitTrainMRRScore));
            Console.WriteLine(String.Format("validation MRR score: {0:0.0000}", logitTestMRRScore));

            // 2. Train a Gaussian SVM Classifier
            Console.WriteLine("\n---- Gaussian SVM Classifier ----\n");
            var teacher = new MulticlassSupportVectorLearning<Gaussian>()
            {
                Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
                {
                    Epsilon = 2,
                    Tolerance = 1e-2,
                    Complexity = 1000,
                    UseKernelEstimation = true
                }
            };
            // Train SVM model using the same train set that was used for Logistic Regression Classifier
            var svmTrainedModel = teacher.Learn(
                input.Where((x,i) => trainSetIDX.Contains(i)).ToArray(),
                output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()
            );

            // Get in-sample & out-of-sample predictions and prediction probabilities for each class
            double[][] svmTrainProbabilities = new double[trainSetIDX.Length][];
            int[] svmTrainPreds = new int[trainSetIDX.Length];
            for (int i = 0; i < trainSetIDX.Length; i++)
            {
                svmTrainPreds[i] = svmTrainedModel.Decide(input[trainSetIDX[i]]);
                svmTrainProbabilities[i] = svmTrainedModel.Probabilities(input[trainSetIDX[i]]);
            }

            double[][] svmTestProbabilities = new double[testSetIDX.Length][];
            int[] svmTestPreds = new int[testSetIDX.Length];
            for (int i = 0; i < testSetIDX.Length; i++)
            {
                svmTestPreds[i] = svmTrainedModel.Decide(input[testSetIDX[i]]);
                svmTestProbabilities[i] = svmTrainedModel.Probabilities(input[testSetIDX[i]]);
            }

            Console.WriteLine(
                String.Format(
                    "train accuracy: {0:0.0000}",  
                    1 - new ZeroOneLoss(output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()).Loss(svmTrainPreds)
                )
            );
            Console.WriteLine(
                String.Format(
                    "validation accuracy: {0:0.0000}", 
                    1 - new ZeroOneLoss(output.Where((x, i) => testSetIDX.Contains(i)).ToArray()).Loss(svmTestPreds)
                )
            );

            // Build confusion matrix
            string[] svmConfMatrix = BuildConfusionMatrix(
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray(), svmTestPreds, 8
            );

            System.IO.File.WriteAllLines(Path.Combine(dataDirPath, "svm-conf-matrix.csv"), svmConfMatrix);

            // Calculate evaluation metrics
            int[][] svmTrainPredRanks = GetPredictionRanks(svmTrainProbabilities);
            int[][] svmTestPredRanks = GetPredictionRanks(svmTestProbabilities);

            double svmTrainMRRScore = ComputeMeanReciprocalRank(
                svmTrainPredRanks,
                output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()
            );
            double svmTestMRRScore = ComputeMeanReciprocalRank(
                svmTestPredRanks,
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray()
            );

            Console.WriteLine("\n---- Gaussian SVM Classifier ----\n");
            Console.WriteLine(String.Format("train MRR score: {0:0.0000}", svmTrainMRRScore));
            Console.WriteLine(String.Format("validation MRR score: {0:0.0000}", svmTestMRRScore));

            // 3. Train a NaiveBayes Classifier
            Console.WriteLine("\n---- NaiveBayes Classifier ----\n");
            var nbTeacher = new NaiveBayesLearning<NormalDistribution>();

            var nbTrainedModel = nbTeacher.Learn(
                input.Where((x, i) => trainSetIDX.Contains(i)).ToArray(),
                output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()
            );

            // Get in-sample & out-of-sample predictions and prediction probabilities for each class
            double[][] nbTrainProbabilities = new double[trainSetIDX.Length][];
            int[] nbTrainPreds = new int[trainSetIDX.Length];
            for (int i = 0; i < trainSetIDX.Length; i++)
            {
                nbTrainProbabilities[i] = nbTrainedModel.Probabilities(input[trainSetIDX[i]]);
                nbTrainPreds[i] = nbTrainedModel.Decide(input[trainSetIDX[i]]);
            }

            double[][] nbTestProbabilities = new double[testSetIDX.Length][];
            int[] nbTestPreds = new int[testSetIDX.Length];
            for (int i = 0; i < testSetIDX.Length; i++)
            {
                nbTestProbabilities[i] = nbTrainedModel.Probabilities(input[testSetIDX[i]]);
                nbTestPreds[i] = nbTrainedModel.Decide(input[testSetIDX[i]]);
            }

            Console.WriteLine(
                String.Format(
                    "train accuracy: {0:0.0000}",
                    1 - new ZeroOneLoss(output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()).Loss(nbTrainPreds)
                )
            );
            Console.WriteLine(
                String.Format(
                    "validation accuracy: {0:0.0000}",
                    1 - new ZeroOneLoss(output.Where((x, i) => testSetIDX.Contains(i)).ToArray()).Loss(nbTestPreds)
                )
            );

            // Build confusion matrix
            string[] nbConfMatrix = BuildConfusionMatrix(
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray(), nbTestPreds, 8
            );

            System.IO.File.WriteAllLines(Path.Combine(dataDirPath, "nb-conf-matrix.csv"), nbConfMatrix);

            // Calculate evaluation metrics
            int[][] nbTrainPredRanks = GetPredictionRanks(nbTrainProbabilities);
            int[][] nbTestPredRanks = GetPredictionRanks(nbTestProbabilities);

            double nbTrainMRRScore = ComputeMeanReciprocalRank(
                nbTrainPredRanks,
                output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()
            );
            double nbTestMRRScore = ComputeMeanReciprocalRank(
                nbTestPredRanks,
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray()
            );

            Console.WriteLine("\n---- NaiveBayes Classifier ----\n");
            Console.WriteLine(String.Format("train MRR score: {0:0.0000}", nbTrainMRRScore));
            Console.WriteLine(String.Format("validation MRR score: {0:0.0000}", nbTestMRRScore));

            // 4. Ensembling Base Models
            Console.WriteLine("\n-- Building Meta Model --");
            double[][] combinedTrainProbabilities = new double[trainSetIDX.Length][];
            for (int i = 0; i < trainSetIDX.Length; i++)
            {
                List<double> combined = trainProbabilities[i]
                    //.Concat(svmTrainProbabilities[i])
                    .Concat(nbTrainProbabilities[i])
                    .ToList();
                combined.Add(logitTrainPreds[i]);
                //combined.Add(svmTrainPreds[i]);
                combined.Add(nbTrainPreds[i]);

                combinedTrainProbabilities[i] = combined.ToArray();
            }

            double[][] combinedTestProbabilities = new double[testSetIDX.Length][];
            for (int i = 0; i < testSetIDX.Length; i++)
            {
                List<double> combined = testProbabilities[i]
                    //.Concat(svmTestProbabilities[i])
                    .Concat(nbTestProbabilities[i])
                    .ToList();
                combined.Add(logitTestPreds[i]);
                //combined.Add(svmTestPreds[i]);
                combined.Add(nbTestPreds[i]);

                combinedTestProbabilities[i] = combined.ToArray();
            }
            Console.WriteLine("\n* input shape: ({0}, {1})\n", combinedTestProbabilities.Length, combinedTestProbabilities[0].Length);

            // Build meta-model using NaiveBayes Learning Algorithm
            var metaModelTeacher = new NaiveBayesLearning<NormalDistribution>();
            var metamodel = metaModelTeacher.Learn(
                combinedTrainProbabilities, 
                output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()
            );

            // Get in-sample & out-of-sample predictions and prediction probabilities for each class
            double[][] metaTrainProbabilities = new double[trainSetIDX.Length][];
            int[] metamodelTrainPreds = new int[trainSetIDX.Length];
            for (int i = 0; i < trainSetIDX.Length; i++)
            {
                metaTrainProbabilities[i] = metamodel.Probabilities(combinedTrainProbabilities[i]);
                metamodelTrainPreds[i] = metamodel.Decide(combinedTrainProbabilities[i]);
            }

            double[][] metaTestProbabilities = new double[testSetIDX.Length][];
            int[] metamodelTestPreds = new int[testSetIDX.Length];
            for (int i = 0; i < testSetIDX.Length; i++)
            {
                metaTestProbabilities[i] = metamodel.Probabilities(combinedTestProbabilities[i]);
                metamodelTestPreds[i] = metamodel.Decide(combinedTestProbabilities[i]);
            }

            Console.WriteLine("\n---- Meta-Model ----\n");
            Console.WriteLine(
                String.Format(
                    "train accuracy: {0:0.0000}",
                    1 - new ZeroOneLoss(output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()).Loss(metamodelTrainPreds)
                )
            );
            Console.WriteLine(
                String.Format(
                    "validation accuracy: {0:0.0000}",
                    1 - new ZeroOneLoss(output.Where((x, i) => testSetIDX.Contains(i)).ToArray()).Loss(metamodelTestPreds)
                )
            );

            // Build confusion matrix
            string[] metamodelConfMatrix = BuildConfusionMatrix(
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray(), metamodelTestPreds, 8
            );

            System.IO.File.WriteAllLines(Path.Combine(dataDirPath, "metamodel-conf-matrix.csv"), metamodelConfMatrix);

            // Calculate evaluation metrics
            int[][] metaTrainPredRanks = GetPredictionRanks(metaTrainProbabilities);
            int[][] metaTestPredRanks = GetPredictionRanks(metaTestProbabilities);

            double metaTrainMRRScore = ComputeMeanReciprocalRank(
                metaTrainPredRanks,
                output.Where((x, i) => trainSetIDX.Contains(i)).ToArray()
            );
            double metaTestMRRScore = ComputeMeanReciprocalRank(
                metaTestPredRanks,
                output.Where((x, i) => testSetIDX.Contains(i)).ToArray()
            );

            Console.WriteLine("\n---- Meta-Model ----\n");
            Console.WriteLine(String.Format("train MRR score: {0:0.0000}", metaTrainMRRScore));
            Console.WriteLine(String.Format("validation MRR score: {0:0.0000}", metaTestMRRScore));

            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }

        private static int[][] GetPredictionRanks(double[][] predProbabilities)
        {
            int[][] rankOrdered = new int[predProbabilities.Length][];

            for(int i = 0; i< predProbabilities.Length; i++)
            {
                rankOrdered[i] = Matrix.ArgSort<double>(predProbabilities[i]).Reversed();
            }

            return rankOrdered;
        }

        private static double ComputeMeanReciprocalRank(int[][] rankOrderedPreds, int[] actualClasses)
        {
            int num = rankOrderedPreds.Length;
            double reciprocalSum = 0.0;

            for(int i = 0; i < num; i++)
            {
                int predRank = 0;
                for(int j = 0; j < rankOrderedPreds[i].Length; j++)
                {
                    if(rankOrderedPreds[i][j] == actualClasses[i])
                    {
                        predRank = j + 1;
                    }
                }
                reciprocalSum += 1.0 / predRank;
            }

            return reciprocalSum / num;
        }


        private static string[] BuildConfusionMatrix(int[] actual, int[] preds, int numClass)
        {
            int[][] matrix = new int[numClass][];
            for(int i = 0; i < numClass; i++)
            {
                matrix[i] = new int[numClass];
            }

            for(int i = 0; i < actual.Length; i++)
            {
                matrix[actual[i]][preds[i]] += 1;
            }

            string[] lines = new string[numClass];
            for(int i = 0; i < matrix.Length; i++)
            {
                lines[i] = string.Join(",", matrix[i]);
            }

            return lines;
        }
    }
}
