using Accord.Controls;
using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization;
using Accord.Math.Optimization.Losses;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Statistics.Analysis;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Kernels;
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

namespace Modeling
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 60);

            // Read in the Image Features dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.8\input-data";

            // Load the data into a data frame
            string trainDataPath = Path.Combine(dataDirPath, "pca-train.csv");
            Console.WriteLine("Loading {0}\n\n", trainDataPath);
            var trainDF = Frame.ReadCsv(
                trainDataPath,
                hasHeaders: false,
                inferTypes: true
            );

            string testDataPath = Path.Combine(dataDirPath, "pca-test.csv");
            Console.WriteLine("Loading {0}\n\n", testDataPath);
            var testDF = Frame.ReadCsv(
                testDataPath,
                hasHeaders: false,
                inferTypes: true
            );

            string[] colnames = trainDF.ColumnKeys.Select(
                (x, i) => i < trainDF.ColumnKeys.Count() - 1 ? String.Format("component-{0}", i + 1) : "label"
            ).ToArray();

            trainDF.RenameColumns(colnames);
            testDF.RenameColumns(colnames);

            // Capturing 70% of the variance
            string[] featureCols = colnames.Where((x, i) => i <= 90).ToArray();


            double[][] trainInput = BuildJaggedArray(
                trainDF.Columns[featureCols].ToArray2D<double>(), trainDF.RowCount, featureCols.Length
            );
            int[] trainOutput = trainDF.GetColumn<int>("label").ValuesAll.ToArray();

            double[][] testInput = BuildJaggedArray(
                testDF.Columns[featureCols].ToArray2D<double>(), testDF.RowCount, featureCols.Length
            );
            int[] testOutput = testDF.GetColumn<int>("label").ValuesAll.ToArray();

            trainDF = null;
            testDF = null;

            GC.Collect();

            Console.WriteLine("* Train Set Shape: {0}, {1}\n\n", trainInput.Length, trainInput[0].Length);
            Console.WriteLine("* Test Set Shape: {0}, {1}\n\n", testInput.Length, testInput[0].Length);


            Console.WriteLine("\n\n---- Training Neural Network Model ----\n");
            BuildNNModel(trainInput, trainOutput, testInput, testOutput);

            Console.WriteLine("\n\n---- Training Naive Bayes Model ----\n");
            BuildNBModel(trainInput, trainOutput, testInput, testOutput);

            Console.WriteLine("\n\n---- Training Logistic Regression Model ----\n");
            BuildLogitModel(trainInput, trainOutput, testInput, testOutput);


            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }

        private static double[][] BuildJaggedArray(double[,] ary2d, int rowCount, int colCount)
        {
            double[][] matrix = new double[rowCount][];
            for(int i = 0; i < rowCount; i++)
            {
                matrix[i] = new double[colCount];
                for(int j = 0; j < colCount; j++)
                {
                    matrix[i][j] = double.IsNaN(ary2d[i, j]) ? 0.0 : ary2d[i, j];
                }
            }
            return matrix;
        }

        private static void BuildNNModel(double[][] trainInput, int[] trainOutput, double[][] testInput, int[] testOutput)
        {
            double[][] outputs = Accord.Math.Jagged.OneHot(trainOutput);

            var function = new BipolarSigmoidFunction(2);
            var network = new ActivationNetwork(
                new BipolarSigmoidFunction(2), 
                91, 
                20,
                10
            );
            
            var teacher = new LevenbergMarquardtLearning(network);

            Console.WriteLine("\n-- Training Neural Network");
            int numEpoch = 10;
            double error = Double.PositiveInfinity;
            for (int i = 0; i < numEpoch; i++)
            {
                error = teacher.RunEpoch(trainInput, outputs);
                Console.WriteLine("* Epoch {0} - error: {1:0.0000}", i + 1, error);
            }
            Console.WriteLine("");

            List<int> inSamplePredsList = new List<int>();
            for (int i = 0; i < trainInput.Length; i++)
            {
                double[] output = network.Compute(trainInput[i]);
                int pred = output.ToList().IndexOf(output.Max());
                inSamplePredsList.Add(pred);
            }

            List<int> outSamplePredsList = new List<int>();
            for (int i = 0; i < testInput.Length; i++)
            {
                double[] output = network.Compute(testInput[i]);
                int pred = output.ToList().IndexOf(output.Max());
                outSamplePredsList.Add(pred);
            }

            int[] inSamplePreds = inSamplePredsList.ToArray();
            int[] outSamplePreds = outSamplePredsList.ToArray();

            // Accuracy
            double inSampleAccuracy = 1 - new ZeroOneLoss(trainOutput).Loss(inSamplePreds);
            double outSampleAccuracy = 1 - new ZeroOneLoss(testOutput).Loss(outSamplePreds);
            Console.WriteLine("* In-Sample Accuracy: {0:0.0000}", inSampleAccuracy);
            Console.WriteLine("* Out-of-Sample Accuracy: {0:0.0000}", outSampleAccuracy);

            // Build confusion matrix
            int[][] confMatrix = BuildConfusionMatrix(
                testOutput, outSamplePreds, 10
            );
            System.IO.File.WriteAllLines(
                Path.Combine(
                    @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.8\input-data",
                    "nn-conf-matrix.csv"
                ),
                confMatrix.Select(x => String.Join(",", x))
            );

            // Precision Recall
            PrintPrecisionRecall(confMatrix);
            DrawROCCurve(testOutput, outSamplePreds, 10, "NN");
        }

        private static void BuildNBModel(double[][] trainInput, int[] trainOutput, double[][] testInput, int[] testOutput)
        {
            var teacher = new NaiveBayesLearning<NormalDistribution>();
            var nbModel = teacher.Learn(trainInput, trainOutput);

            int[] inSamplePreds = nbModel.Decide(trainInput);
            int[] outSamplePreds = nbModel.Decide(testInput);

            // Accuracy
            double inSampleAccuracy = 1 - new ZeroOneLoss(trainOutput).Loss(inSamplePreds);
            double outSampleAccuracy = 1 - new ZeroOneLoss(testOutput).Loss(outSamplePreds);
            Console.WriteLine("* In-Sample Accuracy: {0:0.0000}", inSampleAccuracy);
            Console.WriteLine("* Out-of-Sample Accuracy: {0:0.0000}", outSampleAccuracy);

            // Build confusion matrix
            int[][] confMatrix = BuildConfusionMatrix(
                testOutput, outSamplePreds, 10
            );
            System.IO.File.WriteAllLines(
                Path.Combine(
                    @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.8\input-data",
                    "nb-conf-matrix.csv"
                ),
                confMatrix.Select(x => String.Join(",", x))
            );

            // Precision Recall
            PrintPrecisionRecall(confMatrix);
            DrawROCCurve(testOutput, outSamplePreds, 10, "NB");
        }

        private static void BuildLogitModel(double[][] trainInput, int[] trainOutput, double[][] testInput, int[] testOutput)
        {
            var logit = new MultinomialLogisticLearning<GradientDescent>()
            {
                MiniBatchSize = 500
            };
            var logitModel = logit.Learn(trainInput, trainOutput);

            int[] inSamplePreds = logitModel.Decide(trainInput);
            int[] outSamplePreds = logitModel.Decide(testInput);

            // Accuracy
            double inSampleAccuracy = 1 - new ZeroOneLoss(trainOutput).Loss(inSamplePreds);
            double outSampleAccuracy = 1 - new ZeroOneLoss(testOutput).Loss(outSamplePreds);
            Console.WriteLine("* In-Sample Accuracy: {0:0.0000}", inSampleAccuracy);
            Console.WriteLine("* Out-of-Sample Accuracy: {0:0.0000}", outSampleAccuracy);

            // Build confusion matrix
            int[][] confMatrix = BuildConfusionMatrix(
                testOutput, outSamplePreds, 10
            );
            System.IO.File.WriteAllLines(
                Path.Combine(
                    @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.8\input-data", 
                    "logit-conf-matrix.csv"
                ),
                confMatrix.Select(x => String.Join(",", x))
            );

            // Precision Recall
            PrintPrecisionRecall(confMatrix);
            DrawROCCurve(testOutput, outSamplePreds, 10, "Logit");
        }

        private static void DrawROCCurve(int[] actual, int[] preds, int numClass, string modelName)
        {
            ScatterplotView spv = new ScatterplotView();
            spv.Dock = DockStyle.Fill;
            spv.LinesVisible = true;

            Color[] colors = new Color[] {
                Color.Blue, Color.Red, Color.Orange, Color.Yellow, Color.Green,
                Color.Gray, Color.LightSalmon, Color.LightSkyBlue, Color.Black, Color.Pink
            };

            for (int i = 0; i < numClass; i++)
            {
                // Build ROC for Train Set
                bool[] expected = actual.Select(x => x == i ? true : false).ToArray();
                int[] predicted = preds.Select(x => x == i ? 1 : 0).ToArray();

                var trainRoc = new ReceiverOperatingCharacteristic(expected, predicted);
                trainRoc.Compute(1000);

                // Get Train AUC
                double auc = trainRoc.Area;
                double[] xVals = trainRoc.Points.Select(x => 1 - x.Specificity).ToArray();
                double[] yVals = trainRoc.Points.Select(x => x.Sensitivity).ToArray();

                // Draw ROC Curve
                spv.Graph.GraphPane.AddCurve(
                    String.Format(
                        "Digit: {0} - AUC: {1:0.00}",
                        i, auc
                    ),
                    xVals, yVals, colors[i], SymbolType.None
                );
                spv.Graph.GraphPane.AxisChange();
            }

            spv.Graph.GraphPane.Title.Text = String.Format(
                "{0} ROC - One vs. Rest",
                modelName
            );

            Form f1 = new Form();
            f1.Width = 700;
            f1.Height = 500;
            f1.Controls.Add(spv);
            f1.ShowDialog();
        }

        private static void PrintPrecisionRecall(int[][] confMatrix)
        {
            for (int i = 0; i < confMatrix.Length; i++)
            {
                int totalActual = confMatrix[i].Sum();
                int correctPredCount = confMatrix[i][i];

                int totalPred = 0;
                for(int j = 0; j < confMatrix.Length; j++)
                {
                    totalPred += confMatrix[j][i];
                }

                double precision = correctPredCount / (float)totalPred;
                double recall = correctPredCount / (float)totalActual;

                Console.WriteLine("- Digit {0}: precision - {1:0.0000}, recall - {2:0.0000}", i, precision, recall);
            }

        }

        private static int[][] BuildConfusionMatrix(int[] actual, int[] preds, int numClass)
        {
            int[][] matrix = new int[numClass][];
            for (int i = 0; i < numClass; i++)
            {
                matrix[i] = new int[numClass];
            }

            for (int i = 0; i < actual.Length; i++)
            {
                matrix[actual[i]][preds[i]] += 1;
            }

            return matrix;
        }
    }
}
