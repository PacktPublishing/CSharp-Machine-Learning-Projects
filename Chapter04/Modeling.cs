using Accord.Controls;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Visualizations;
using Deedle;
using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Modeling
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 50);

            // Read in the file we created in the previous step
            // TODO: change the path to point to your data directory
            string dataDirPath = @"<path-to-your-dir>";

            // Load the data into a data frame
            Console.WriteLine("Loading data...");
            var featuresDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "eurusd-features.csv"),
                hasHeaders: true,
                inferTypes: true
            );

            // Split the sample set into train and test sets
            double trainProportion = 0.9;
            int trainSetIndexMax = (int)(featuresDF.RowCount * trainProportion);

            var trainSet = featuresDF.Where(x => x.Key < trainSetIndexMax);
            var testSet = featuresDF.Where(x => x.Key >= trainSetIndexMax);

            Console.WriteLine("\nTrain Set Shape: ({0}, {1})", trainSet.RowCount, trainSet.ColumnCount);
            Console.WriteLine("Test Set Shape: ({0}, {1})", testSet.RowCount, testSet.ColumnCount);

            string[] features = new string[] {
                "DailyReturn", "Close_minus_10_MA", "Close_minus_20_MA", "Close_minus_50_MA",
                "Close_minus_200_MA", "20_day_std", "Close_minus_BollingerUpperBound",
                "Close_minus_BollingerLowerBound", "DailyReturn_T-1", "DailyReturn_T-2",
                "DailyReturn_T-3", "DailyReturn_T-4", "DailyReturn_T-5",
                "Close_minus_10_MA_T-1",
                "Close_minus_10_MA_T-2", "Close_minus_10_MA_T-3", "Close_minus_10_MA_T-4",
                "Close_minus_10_MA_T-5", "Close_minus_20_MA_T-1", "Close_minus_20_MA_T-2",
                "Close_minus_20_MA_T-3", "Close_minus_20_MA_T-4", "Close_minus_20_MA_T-5",
                "Close_minus_50_MA_T-1", "Close_minus_50_MA_T-2", "Close_minus_50_MA_T-3",
                "Close_minus_50_MA_T-4", "Close_minus_50_MA_T-5", "Close_minus_200_MA_T-1",
                "Close_minus_200_MA_T-2", "Close_minus_200_MA_T-3", "Close_minus_200_MA_T-4",
                "Close_minus_200_MA_T-5",
                "Close_minus_BollingerUpperBound_T-1",
                "Close_minus_BollingerUpperBound_T-2", "Close_minus_BollingerUpperBound_T-3",
                "Close_minus_BollingerUpperBound_T-4", "Close_minus_BollingerUpperBound_T-5"
            };

            double[][] trainX = BuildJaggedArray(
                trainSet.Columns[features].ToArray2D<double>(), 
                trainSet.RowCount,
                features.Length
            );
            double[][] testX = BuildJaggedArray(
                testSet.Columns[features].ToArray2D<double>(),
                testSet.RowCount,
                features.Length
            );

            double[] trainY = trainSet["Target"].ValuesAll.ToArray();
            double[] testY = testSet["Target"].ValuesAll.ToArray();

            Console.WriteLine("\n**** Linear Regression Model ****");

            // OLS learning algorithm
            var ols = new OrdinaryLeastSquares()
            {
                UseIntercept = true
            };

            // Fit a linear regression model
            MultipleLinearRegression regFit = ols.Learn(trainX, trainY);

            // in-sample predictions
            double[] regInSamplePreds = regFit.Transform(trainX);
            // out-of-sample predictions
            double[] regOutSamplePreds = regFit.Transform(testX);

            ValidateModelResults("Linear Regression", regInSamplePreds, regOutSamplePreds, trainX, trainY, testX, testY);

            Console.WriteLine("\n* Linear Regression Coefficients:");
            for (int i = 0; i < features.Length; i++)
            {
                Console.WriteLine("\t{0}: {1:0.0000}", features[i], regFit.Weights[i]);
            }

            Console.WriteLine("\tIntercept: {0:0.0000}", regFit.Intercept);


            Console.WriteLine("\n**** Linear Support Vector Machine ****");
            // Linear SVM Learning Algorithm
            var teacher = new LinearRegressionNewtonMethod()
            {
                Epsilon = 2.1,
                Tolerance = 1e-5,
                UseComplexityHeuristic = true
            };

            // Train SVM
            var svm = teacher.Learn(trainX, trainY);

            // in-sample predictions
            double[] linSVMInSamplePreds = svm.Score(trainX);
            // out-of-sample predictions
            double[] linSVMOutSamplePreds = svm.Score(testX);

            ValidateModelResults("Linear SVM", linSVMInSamplePreds, linSVMOutSamplePreds, trainX, trainY, testX, testY);

            Console.WriteLine("\n\n\nDONE!!");
            Console.ReadKey();
        }

        private static void ValidateModelResults(string modelName, double[] regInSamplePreds, double[] regOutSamplePreds, double[][] trainX, double[] trainY, double[][] testX, double[] testY)
        {
            // RMSE for in-sample 
            double regInSampleRMSE = Math.Sqrt(new SquareLoss(trainX).Loss(regInSamplePreds));
            // RMSE for out-sample 
            double regOutSampleRMSE = Math.Sqrt(new SquareLoss(testX).Loss(regOutSamplePreds));

            Console.WriteLine("RMSE: {0:0.0000} (Train) vs. {1:0.0000} (Test)", regInSampleRMSE, regOutSampleRMSE);

            // R^2 for in-sample 
            double regInSampleR2 = new RSquaredLoss(trainX[0].Length, trainX).Loss(regInSamplePreds);
            // R^2 for out-sample 
            double regOutSampleR2 = new RSquaredLoss(testX[0].Length, testX).Loss(regOutSamplePreds);

            Console.WriteLine("R^2: {0:0.0000} (Train) vs. {1:0.0000} (Test)", regInSampleR2, regOutSampleR2);

            // Scatter Plot of expected and actual
            ScatterplotBox.Show(
                String.Format("Actual vs. Prediction ({0})", modelName), testY, regOutSamplePreds
            );
        }

        private static double[][] BuildJaggedArray(double[,] ary2D, int rowCount, int columnCount)
        {
            double[][] ary = new double[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                ary[i] = new double[columnCount];
                for (int j = 0; j < columnCount; j++)
                {
                    ary[i][j] = double.IsNaN(ary2D[i, j])? 0.0: ary2D[i, j];
                }
            }
            return ary;
        }
    }
}
