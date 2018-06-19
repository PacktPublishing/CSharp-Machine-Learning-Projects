using Accord.Controls;
using Accord.MachineLearning;
using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.Performance;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Analysis;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using Deedle;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ZedGraph;

namespace TwitterSentimentModeling
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Console.SetWindowSize(250, 80);

            // Read in the file we created in the previous step
            // TODO: change the path to point to your data directory
            string dataDirPath = @"<path-to-your-data-dir>";

            // Load the data into a data frame
            Console.WriteLine("Loading data...");
            var lemmaVecDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "tweet-lemma.csv"),
                hasHeaders: true,
                inferTypes: true
            );

            // Load Term Frequency Data
            Console.WriteLine("Loading Term Frequencies...");
            var positiveTermFrequencyDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "positive-frequencies.csv"),
                hasHeaders: false,
                inferTypes: false,
                schema: "string,int"
            );
            positiveTermFrequencyDF.RenameColumns(new string[] { "term", "count" });
            var indexedPositiveTermFrequencyDF = positiveTermFrequencyDF.IndexRows<string>("term");

            var negativeTermFrequencyDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "negative-frequencies.csv"),
                hasHeaders: false,
                inferTypes: false,
                schema: "string,int"
            );
            negativeTermFrequencyDF.RenameColumns(new string[] { "term", "count" });
            var indexedNegativeTermFrequencyDF = negativeTermFrequencyDF.IndexRows<string>("term");

            // Change number of features to reduce overfitting
            int[] featureSelections = new int[] { 5, 10, 50, 100, 150 };
            foreach(int minNumOccurences in featureSelections)
            {
                Console.WriteLine("\n\n---- Starting a new Model Building Process ----");
                string[] termFeatures = new HashSet<string>(
                indexedPositiveTermFrequencyDF.Where(
                        x => x.Value.GetAs<int>("count") >= minNumOccurences
                    ).RowKeys
                ).Union(
                    new HashSet<string>(
                        indexedNegativeTermFrequencyDF.Where(
                            x => x.Value.GetAs<int>("count") >= minNumOccurences
                        ).RowKeys
                    )
                ).ToArray();
                Console.WriteLine("* Num Features Selected: {0} (# Occurences >= {1})", termFeatures.Count(), minNumOccurences);

                // get sentiment target veriable
                var targetVariables = lemmaVecDF.GetColumn<int>("tweet_polarity");

                var sampleSetDistribution = targetVariables.GroupBy<int>(x => x.Value).Select(x => x.Value.KeyCount);
                int[] sampleSizes = sampleSetDistribution.Values.ToArray();
                Console.WriteLine(
                    "* Sentiment Distribution: {0} neutral vs. {1} positive vs. {2} negative",
                    sampleSizes[0], sampleSizes[1], sampleSizes[2]
                );

                // Create input and output variables from data frames, so that we can use them for Accord.NET MachineLearning models
                double[][] input = lemmaVecDF.Columns[termFeatures].Rows.Select(
                    x => Array.ConvertAll<object, double>(x.Value.ValuesAll.ToArray(), o => Convert.ToDouble(o))
                ).ValuesAll.ToArray();
                int[] output = targetVariables.Values.ToArray();

                // Split the sample set into Train (80%) and Test (20%) sets and Train a NaiveBayes Classifier
                Console.WriteLine("\n---- Training NaiveBayes Classifier ----");
                var nbSplitSet = new SplitSetValidation<NaiveBayes<BernoulliDistribution>, double[]>()
                {
                   Learner = (s) => new NaiveBayesLearning<BernoulliDistribution>(),

                   Loss = (expected, actual, p) => new ZeroOneLoss(expected).Loss(actual),

                   Stratify = false,

                   TrainingSetProportion = 0.8,

                   ValidationSetProportion = 0.2
                };
                var nbResult = nbSplitSet.Learn(input, output);

                // Get in-sample & out-sample prediction results for NaiveBayes Classifier
                var nbTrainedModel = nbResult.Model;

                int[] nbTrainSetIDX = nbSplitSet.IndicesTrainingSet;
                int[] nbTestSetIDX = nbSplitSet.IndicesValidationSet;

                Console.WriteLine("* Train Set Size: {0}, Test Set Size: {1}", nbTrainSetIDX.Length, nbTestSetIDX.Length);

                int[] nbTrainPreds = new int[nbTrainSetIDX.Length];
                int[] nbTrainActual = new int[nbTrainSetIDX.Length];
                for (int i = 0; i < nbTrainPreds.Length; i++)
                {
                   nbTrainActual[i] = output[nbTrainSetIDX[i]];
                   nbTrainPreds[i] = nbTrainedModel.Decide(input[nbTrainSetIDX[i]]);
                }

                int[] nbTestPreds = new int[nbTestSetIDX.Length];
                int[] nbTestActual = new int[nbTestSetIDX.Length];
                for (int i = 0; i < nbTestPreds.Length; i++)
                {
                   nbTestActual[i] = output[nbTestSetIDX[i]];
                   nbTestPreds[i] = nbTrainedModel.Decide(input[nbTestSetIDX[i]]);
                }

                // Evaluate NaiveBayes Model Performance
                PrintConfusionMatrix(nbTrainPreds, nbTrainActual, nbTestPreds, nbTestActual);
                DrawROCCurve(nbTrainActual, nbTrainPreds, nbTestActual, nbTestPreds, 0, minNumOccurences, "NaiveBayes");
                DrawROCCurve(nbTrainActual, nbTrainPreds, nbTestActual, nbTestPreds, 1, minNumOccurences, "NaiveBayes");
                DrawROCCurve(nbTrainActual, nbTrainPreds, nbTestActual, nbTestPreds, 2, minNumOccurences, "NaiveBayes");

                // Split the sample set into Train (80%) and Test (20%) sets and Train a RandomForest Classifier
                Console.WriteLine("\n---- Training RandomForest Classifier ----");
                var rfSplitSet = new SplitSetValidation<RandomForest, double[]>()
                {
                    Learner = (s) => new RandomForestLearning()
                    {
                        NumberOfTrees = 100, // Change this hyperparameter for further tuning

                        CoverageRatio = 0.5, // the proportion of variables that can be used at maximum by each tree

                        SampleRatio = 0.7 // the proportion of samples used to train each of the trees

                    },

                    Loss = (expected, actual, p) => new ZeroOneLoss(expected).Loss(actual),

                    Stratify = false,

                    TrainingSetProportion = 0.7,

                    ValidationSetProportion = 0.3
                };
                var rfResult = rfSplitSet.Learn(input, output);

                // Get in-sample & out-sample prediction results for RandomForest Classifier
                var rfTrainedModel = rfResult.Model;

                int[] rfTrainSetIDX = rfSplitSet.IndicesTrainingSet;
                int[] rfTestSetIDX = rfSplitSet.IndicesValidationSet;

                Console.WriteLine("* Train Set Size: {0}, Test Set Size: {1}", rfTrainSetIDX.Length, rfTestSetIDX.Length);

                int[] rfTrainPreds = new int[rfTrainSetIDX.Length];
                int[] rfTrainActual = new int[rfTrainSetIDX.Length];
                for (int i = 0; i < rfTrainPreds.Length; i++)
                {
                    rfTrainActual[i] = output[rfTrainSetIDX[i]];
                    rfTrainPreds[i] = rfTrainedModel.Decide(input[rfTrainSetIDX[i]]);
                }

                int[] rfTestPreds = new int[rfTestSetIDX.Length];
                int[] rfTestActual = new int[rfTestSetIDX.Length];
                for (int i = 0; i < rfTestPreds.Length; i++)
                {
                    rfTestActual[i] = output[rfTestSetIDX[i]];
                    rfTestPreds[i] = rfTrainedModel.Decide(input[rfTestSetIDX[i]]);
                }

                // Evaluate RandomForest Model Performance
                PrintConfusionMatrix(rfTrainPreds, rfTrainActual, rfTestPreds, rfTestActual);
                Console.WriteLine("");
                DrawROCCurve(rfTrainActual, rfTrainPreds, rfTestActual, rfTestPreds, 0, minNumOccurences, "RandomForest");
                DrawROCCurve(rfTrainActual, rfTrainPreds, rfTestActual, rfTestPreds, 1, minNumOccurences, "RandomForest");
                DrawROCCurve(rfTrainActual, rfTrainPreds, rfTestActual, rfTestPreds, 2, minNumOccurences, "RandomForest");
            }

            Console.ReadKey();
        }

        private static void PrintConfusionMatrix(int[] trainPreds, int[] trainActual, int[] testPreds, int[] testActual)
        {
            GeneralConfusionMatrix trainCM = new GeneralConfusionMatrix(trainPreds, trainActual);
            GeneralConfusionMatrix testCM = new GeneralConfusionMatrix(testPreds, testActual);

            // Print Train Confusion Matrix
            Console.WriteLine("\n---- Train Set Confusion Matrix ----");
            PrintConfusionMatrix(trainCM);

            Console.WriteLine("\n---- Test Set Confusion Matrix ----");
            PrintConfusionMatrix(testCM);
        }

        private static void PrintConfusionMatrix(GeneralConfusionMatrix cm)
        {
            int numberOfSamples = 0;
            Console.Write("\t\tActual 0\t\tActual 1\t\tActual 2\n");
            for (int i = 0; i < cm.Matrix.GetLength(0); i++)
            {
                Console.Write("Pred {0} :\t", i);
                for (int j = 0; j < cm.Matrix.GetLength(1); j++)
                {
                    int count = cm.Matrix[i, j];
                    numberOfSamples += count;
                    Console.Write(count + "\t\t\t");
                }
                Console.WriteLine();
            }

            Console.WriteLine("\n---- Calculating Accuracy, Precision, Recall ----");

            float trueNeutral = (float)cm.Matrix[0, 0];
            float truePositive = (float)cm.Matrix[1, 1];
            float trueNegative = (float)cm.Matrix[2, 2];

            // Accuracy
            Console.WriteLine(
                "-- Accuracy: {0:0.00}%",
                (trueNeutral + truePositive + trueNegative) / numberOfSamples * 100.0
            );
            // Precision vs. Recall
            Console.WriteLine(
                "-- Neutral Class Precision: {0:0.00}% vs. Recall: {1:0.00}%", 
                (trueNeutral / (trueNeutral + (float)cm.Matrix[0, 1] + (float)cm.Matrix[0, 2])) * 100.0,
                (trueNeutral / (trueNeutral + (float)cm.Matrix[1, 0] + (float)cm.Matrix[2, 0])) * 100.0
            );
            Console.WriteLine(
                "-- Positive Class Precision: {0:0.00}% vs. Recall: {1:0.00}%", 
                (truePositive / (truePositive + (float)cm.Matrix[1, 0] + (float)cm.Matrix[1, 2])) * 100.0,
                (truePositive / (truePositive + (float)cm.Matrix[0, 1] + (float)cm.Matrix[2, 1])) * 100.0
            );
            Console.WriteLine(
                "-- Negative Class Precision: {0:0.00}% vs. Recall: {1:0.00}%", 
                (trueNegative / (trueNegative + (float)cm.Matrix[2, 0] + (float)cm.Matrix[2, 1])) * 100.0,
                (trueNegative / (trueNegative + (float)cm.Matrix[0, 2] + (float)cm.Matrix[1, 2])) * 100.0
            );
        }

        private static void DrawROCCurve(int[] trainActual, int[] trainPreds, int[] testActual, int[] testPreds, int predClass, int minNumOccurrences, string modelName)
        {
            // Create a new ROC curve to assess the performance of the model
            string predClassStr = predClass == 0 ? "Neutral" : predClass == 1 ? "Positive" : "Negative";
            Console.WriteLine(
                "* Building ROC curve for {0} vs. Rest",
                predClassStr
            );

            // Build ROC for Train Set
            bool[] trainExpectedClass = trainActual.Select(x => x == predClass ? true : false).ToArray();
            int[] trainPredictedClass = trainPreds.Select(x => x == predClass ? 1 : 0).ToArray();

            var trainRoc = new ReceiverOperatingCharacteristic(trainExpectedClass, trainPredictedClass);
            trainRoc.Compute(1000);

            // Get Train AUC
            double trainAUC = trainRoc.Area;
            double[] trainXValues = trainRoc.Points.Select(x => 1 - x.Specificity).ToArray();
            double[] trainYValues = trainRoc.Points.Select(x => x.Sensitivity).ToArray();

            // Build ROC for Test Set
            bool[] testExpectedClass = testActual.Select(x => x == predClass ? true : false).ToArray();
            int[] testPredictedClass = testPreds.Select(x => x == predClass ? 1 : 0).ToArray();

            var testRoc = new ReceiverOperatingCharacteristic(testExpectedClass, testPredictedClass);
            testRoc.Compute(1000);

            // Get Test AUC
            double testAUC = testRoc.Area;
            double[] testXValues = testRoc.Points.Select(x => 1 - x.Specificity).ToArray();
            double[] testYValues = testRoc.Points.Select(x => x.Sensitivity).ToArray();

            // Draw ROC Curve with both Train & Test ROC
            ScatterplotView spv = new ScatterplotView();
            spv.Dock = DockStyle.Fill;
            spv.LinesVisible = true;
            
            spv.Graph.GraphPane.AddCurve(
                String.Format("Train (AUC: {0:0.00})", trainAUC),
                trainXValues, trainYValues, Color.Green, SymbolType.None
            );
            spv.Graph.GraphPane.AddCurve(
                String.Format("Test (AUC: {0:0.00})", testAUC),
                testXValues, testYValues, Color.Blue, SymbolType.None
            );
            spv.Graph.GraphPane.AddCurve("Random", testXValues, testXValues, Color.Red, SymbolType.None);

            spv.Graph.GraphPane.Title.Text = String.Format(
                "{0} ROC - {1} vs. Rest (# occurrences >= {2})", 
                modelName, predClassStr, minNumOccurrences
            );
            spv.Graph.GraphPane.AxisChange();

            Form f1 = new Form();
            f1.Width = 700;
            f1.Height = 500;
            f1.Controls.Add(spv);
            f1.ShowDialog();
        }
    }
}
