using Accord.MachineLearning;
using Accord.MachineLearning.Bayes;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Analysis;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelBuilder
{
    class Program
    {
        static void Main(string[] args)
        {
            // Read in the file we created in the Data Preparation step
            // TODO: change the path to point to your data directory
            string dataDirPath = "\\\\Mac\\Home\\Documents\\c-sharp-machine-learning\\ch.2\\output";
            // Load the data into a data frame and set the "emailNum" column as an index
            var wordVecDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "data-preparation-step\\subjectWordVec-alphaonly.csv"),
                hasHeaders: true,
                inferTypes: true
            );
            // Load the transformed data from data preparation step to get "is_ham" column
            var rawDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "data-preparation-step\\transformed.csv"),
                hasHeaders: true,
                inferTypes: false,
                schema: "int,string,string,int"
            ).IndexRows<int>("emailNum").SortRowsByKey();
            // Load Term Frequency Data
            var spamTermFrequencyDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "data-analysis-step\\frequency-alphaonly\\subject-line\\spam-frequencies-after-stopwords.csv"),
                hasHeaders: false,
                inferTypes: false,
                schema: "string,int"
            );
            spamTermFrequencyDF.RenameColumns(new string[] { "word", "num_occurences" });
            var indexedSpamTermFrequencyDF = spamTermFrequencyDF.IndexRows<string>("word");

            // Change number of features to reduce overfitting
            int minNumOccurences = 1;
            string[] wordFeatures = indexedSpamTermFrequencyDF.Where(
                x => x.Value.GetAs<int>("num_occurences") >= minNumOccurences
            ).RowKeys.ToArray();
            Console.WriteLine("Num Features Selected: {0}", wordFeatures.Count());

            // subtracting "is_ham" values from 1 to encode this target variable with 1 for spam emails 
            var targetVariables = 1 - rawDF.GetColumn<int>("is_ham");
            Console.WriteLine("{0} spams vs. {1} hams", targetVariables.NumSum(), (targetVariables.KeyCount - targetVariables.NumSum()));

            // Create input and output variables from data frames, so that we can use them for Accord.NET MachineLearning models
            double[][] input = wordVecDF.Columns[wordFeatures].Rows.Select(
                x => Array.ConvertAll<object, double>(x.Value.ValuesAll.ToArray(), o => Convert.ToDouble(o))
            ).ValuesAll.ToArray();
            int[] output = targetVariables.Values.ToArray();

            // Number of folds
            int numFolds = 3;

            var cvNaiveBayesClassifier = CrossValidation.Create<NaiveBayes<BernoulliDistribution>, NaiveBayesLearning<BernoulliDistribution>, double[], int>(
                // number of folds
                k: numFolds,
                // Naive Bayes Classifier with Binomial Distribution
                learner: (p) => new NaiveBayesLearning<BernoulliDistribution>(),
                // Using Zero-One Loss Function as a Cost Function
                loss: (actual, expected, p) => new ZeroOneLoss(expected).Loss(actual),
                // Fitting a classifier
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),
                // Input with Features
                x: input,
                // Output
                y: output
            );

            // Run Cross-Validation
            var result = cvNaiveBayesClassifier.Learn(input, output);

            // Sample Size
            int numberOfSamples = result.NumberOfSamples;
            int numberOfInputs = result.NumberOfInputs;
            int numberOfOutputs = result.NumberOfOutputs;

            // Training & Validation Errors
            double trainingError = result.Training.Mean;
            double validationError = result.Validation.Mean;

            // Confusion Matrix
            Console.WriteLine("\n---- Confusion Matrix ----");
            GeneralConfusionMatrix gcm = result.ToConfusionMatrix(input, output);
            Console.WriteLine("");
            Console.Write("\t\tActual 0\t\tActual 1\n");
            for (int i = 0; i < gcm.Matrix.GetLength(0); i++)
            {
                Console.Write("Pred {0} :\t", i);
                for (int j = 0; j < gcm.Matrix.GetLength(1); j++)
                {
                    Console.Write(gcm.Matrix[i, j] + "\t\t\t");
                }
                Console.WriteLine();
            }

            Console.WriteLine("\n---- Sample Size ----");
            Console.WriteLine("# samples: {0}, # inputs: {1}, # outputs: {2}", numberOfSamples, numberOfInputs, numberOfOutputs);
            Console.WriteLine("training error: {0}", trainingError);
            Console.WriteLine("validation error: {0}\n", validationError);

            Console.WriteLine("\n---- Calculating Accuracy, Precision, Recall ----");

            float truePositive = (float)gcm.Matrix[1, 1];
            float trueNegative = (float)gcm.Matrix[0, 0];
            float falsePositive = (float)gcm.Matrix[1, 0];
            float falseNegative = (float)gcm.Matrix[0, 1];

            // Accuracy
            Console.WriteLine(
                "Accuracy: {0}",
                (truePositive + trueNegative) / numberOfSamples
            );
            // True-Positive / (True-Positive + False-Positive)
            Console.WriteLine("Precision: {0}", (truePositive / (truePositive + falsePositive)));
            // True-Positive / (True-Positive + False-Negative)
            Console.WriteLine("Recall: {0}", (truePositive / (truePositive + falseNegative)));

            Console.ReadKey();
        }
    }
}
