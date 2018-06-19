using Accord.Controls;
using Accord.MachineLearning;
using Accord.MachineLearning.Performance;
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
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.5\input-data";

            // Load the data into a data frame
            Console.WriteLine("Loading data...");
            var featuresDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "features.csv"),
                hasHeaders: true,
                inferTypes: true
            ).FillMissing(0.0);

            // Split the sample set into train and test sets
            double trainProportion = 0.8;

            int[] shuffledIndexes = featuresDF.RowKeys.ToArray();
            shuffledIndexes.Shuffle();

            int trainSetIndexMax = (int)(featuresDF.RowCount * trainProportion);
            int[] trainIndexes = shuffledIndexes.Where(i => i < trainSetIndexMax).ToArray();
            int[] testIndexes = shuffledIndexes.Where(i => i >= trainSetIndexMax).ToArray();

            var trainSet = featuresDF.Where(x => trainIndexes.Contains(x.Key));
            var testSet = featuresDF.Where(x => testIndexes.Contains(x.Key));

            Console.WriteLine("\nTrain Set Shape: ({0}, {1})", trainSet.RowCount, trainSet.ColumnCount);
            Console.WriteLine("Test Set Shape: ({0}, {1})", testSet.RowCount, testSet.ColumnCount);

            string targetVar = "LogSalePrice";
            string[] features = featuresDF.ColumnKeys.Where(
                x => !x.Equals("Id") && !x.Equals(targetVar) && !x.Equals("SalePrice")
            ).ToArray();

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

            double[] trainY = trainSet[targetVar].ValuesAll.ToArray();
            double[] testY = testSet[targetVar].ValuesAll.ToArray();

            Console.WriteLine("\n**** Linear Regression Model ****");

            // OLS learning algorithm
            var ols = new OrdinaryLeastSquares()
            {
                UseIntercept = true,
                IsRobust = true
            };

            // Fit a linear regression model
            MultipleLinearRegression regFit = ols.Learn(
                trainX,
                trainY
            );

            // in-sample predictions
            double[] regInSamplePreds = regFit.Transform(trainX);
            // out-of-sample predictions
            double[] regOutSamplePreds = regFit.Transform(testX);

            ValidateModelResults("Linear Regression", regInSamplePreds, regOutSamplePreds, trainX, trainY, testX, testY);

            //Console.WriteLine("\n* Linear Regression Coefficients:");
            //for (int i = 0; i < features.Length; i++)
            //{
            //    Console.WriteLine("\t{0}: {1:0.0000}", features[i], regFit.Weights[i]);
            //}

            //Console.WriteLine("\tIntercept: {0:0.0000}", regFit.Intercept);


            Console.WriteLine("\n**** Linear Support Vector Machine ****");
            // Linear SVM Learning Algorithm
            var teacher = new LinearRegressionNewtonMethod()
            {
                Epsilon = 0.5,
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

            Console.WriteLine("\n**** Support Vector Machine with Polynomial Kernel ****");
            // SVM with Polynomial Kernel
            var polySVMLearner = new FanChenLinSupportVectorRegression<Polynomial>()
            {
                Epsilon = 0.1,
                Tolerance = 1e-5,
                UseKernelEstimation = true,
                UseComplexityHeuristic = true,
                Kernel = new Polynomial(3)
            };

            // Train SVM with Polynomial Kernel
            var polySvm = polySVMLearner.Learn(trainX, trainY);

            // in-sample predictions
            double[] polySVMInSamplePreds = polySvm.Score(trainX);
            // out-of-sample predictions
            double[] polySVMOutSamplePreds = polySvm.Score(testX);

            ValidateModelResults("Polynomial SVM", polySVMInSamplePreds, polySVMOutSamplePreds, trainX, trainY, testX, testY);


            Console.WriteLine("\n**** Support Vector Machine with Gaussian Kernel ****");
            // SVM with Gaussian Kernel
            var gaussianSVMLearner = new FanChenLinSupportVectorRegression<Gaussian>()
            {
                Epsilon = 0.1,
                Tolerance = 1e-5,
                Complexity = 1e-4,
                UseKernelEstimation = true,
                Kernel = new Gaussian()
            };

            // Train SVM with Gaussian Kernel
            var gaussianSvm = gaussianSVMLearner.Learn(trainX, trainY);

            // in-sample predictions
            double[] guassianSVMInSamplePreds = gaussianSvm.Score(trainX);
            // out-of-sample predictions
            double[] guassianSVMOutSamplePreds = gaussianSvm.Score(testX);

            ValidateModelResults("Guassian SVM", guassianSVMInSamplePreds, guassianSVMOutSamplePreds, trainX, trainY, testX, testY);


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
            var scatterplot = ScatterplotBox.Show(
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
                    ary[i][j] = double.IsNaN(ary2D[i, j]) ? 0.0 : ary2D[i, j];
                }
            }
            return ary;
        }
    }
}
