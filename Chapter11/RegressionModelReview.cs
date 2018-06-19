using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression.Linear;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RegressionModelReview
{
    class Program
    {
        static void Main(string[] args)
        {
            // sample input and output
            double[] inputs = { 10, 20, 30, 40, 50 };
            double[] outputs = { 1, 2, 3, 4, 5 };

            // 1. Linear Regression
            var learner = new OrdinaryLeastSquares()
            {
                UseIntercept = true
            };
            var model = learner.Learn(inputs, outputs);
            var preds = model.Transform(inputs);

            Console.WriteLine("\n\n* Linear Regression Preds: {0}", String.Join(", ", preds));

            // 2. Linear SVM
            var learner2 = new LinearRegressionNewtonMethod()
            {
                Epsilon = 2.1,
                Tolerance = 1e-5,
                UseComplexityHeuristic = true
            };

            var svmInputs = inputs.Select(x => new double[] { x, x }).ToArray();

            var model2 = learner2.Learn(svmInputs, outputs);
            var preds2 = model2.Score(svmInputs);

            Console.WriteLine("\n\n* Linear SVM Preds: {0}", String.Join(", ", preds2));

            // 3. Polynomial SVM
            var learner3 = new FanChenLinSupportVectorRegression<Polynomial>()
            {
                Kernel = new Polynomial(3)
            };
            var model3 = learner3.Learn(svmInputs, outputs);

            var preds3 = model3.Score(svmInputs);

            Console.WriteLine("\n\n* Polynomial SVM Preds: {0}", String.Join(", ", preds3));

            // 4. Gaussian SVM
            var learner4 = new FanChenLinSupportVectorRegression<Gaussian>()
            {
                Kernel = new Gaussian()
            };
            var model4 = learner4.Learn(svmInputs, outputs);

            var preds4 = model4.Score(svmInputs);

            Console.WriteLine("\n\n* Gaussian SVM Preds: {0}", String.Join(", ", preds4));


            Console.WriteLine("\n\n\n\nDONE!!");
            Console.ReadKey();
        }
    }
}
