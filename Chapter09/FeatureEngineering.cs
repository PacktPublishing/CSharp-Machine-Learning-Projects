using Accord.Controls;
using Accord.Statistics.Analysis;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeatureEngineering
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 60);

            // Read in the Cyber Attack dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.9\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "data.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var rawDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            // Encode Categorical Variables
            string[] categoricalVars =
            {
                "protocol_type", "service", "flag", "land"
            };
            // Encode Target Variables
            IDictionary<string, int> targetVarEncoding = new Dictionary<string, int>
            {
                {"normal", 0},
                {"dos", 1},
                {"probe", 2},
                {"r2l", 3},
                {"u2r", 4}
            };

            var featuresDF = Frame.CreateEmpty<int, string>();

            foreach (string col in rawDF.ColumnKeys)
            {
                if(col.Equals("attack_type"))
                {
                    continue;
                }
                else if (col.Equals("attack_category"))
                {
                    featuresDF.AddColumn(
                        col, 
                        rawDF.GetColumn<string>(col).Select(x => targetVarEncoding[x.Value])
                    );
                }
                else if (categoricalVars.Contains(col))
                {
                    var categoryDF = EncodeOneHot(rawDF.GetColumn<string>(col), col);

                    foreach (string newCol in categoryDF.ColumnKeys)
                    {
                        featuresDF.AddColumn(newCol, categoryDF.GetColumn<int>(newCol));
                    }
                }
                else
                {
                    featuresDF.AddColumn(
                        col, 
                        rawDF[col].Select((x, i) => double.IsNaN(x.Value) ? 0.0 : x.Value)
                    );
                }
            }
            Console.WriteLine("* Shape: {0}, {1}\n\n", featuresDF.RowCount, featuresDF.ColumnCount);
            Console.WriteLine("* Exporting feature set...");
            featuresDF.SaveCsv(Path.Combine(dataDirPath, "features.csv"));

            // Build PCA with only normal data
            var rnd = new Random();

            int[] normalIdx = featuresDF["attack_category"]
                .Where(x => x.Value == 0)
                .Keys
                .OrderBy(x => rnd.Next())
                .Take(90000).ToArray();
            int[] attackIdx = featuresDF["attack_category"]
                .Where(x => x.Value > 0)
                .Keys
                .OrderBy(x => rnd.Next())
                .Take(10000).ToArray();
            int[] totalIdx = normalIdx.Concat(attackIdx).ToArray();

            var normalSet = featuresDF.Rows[normalIdx];

            string[] nonZeroValueCols = normalSet.ColumnKeys.Where(
                x =>  !x.Equals("attack_category") && normalSet[x].Max() != normalSet[x].Min()
            ).ToArray();

            double[][] normalData = BuildJaggedArray(
                normalSet.Columns[nonZeroValueCols].ToArray2D<double>(), 
                normalSet.RowCount, 
                nonZeroValueCols.Length
            );
            double[][] wholeData = BuildJaggedArray(
                featuresDF.Rows[totalIdx].Columns[nonZeroValueCols].ToArray2D<double>(),
                totalIdx.Length,
                nonZeroValueCols.Length
            );
            int[] labels = featuresDF
                .Rows[totalIdx]
                .GetColumn<int>("attack_category")
                .ValuesAll.ToArray();

            var pca = new PrincipalComponentAnalysis(
                PrincipalComponentMethod.Standardize
            );
            pca.Learn(normalData);

            double[][] transformed = pca.Transform(wholeData);
            double[][] first2Components = transformed.Select(
                x => x.Where((y, i) => i < 2).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #1 vs. Component #2", first2Components, labels);
            double[][] next2Components = transformed.Select(
                x => x.Where((y, i) => i < 3 && i >= 1).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #2 vs. Component #3", next2Components, labels);
            next2Components = transformed.Select(
                x => x.Where((y, i) => i < 4 && i >= 2).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #3 vs. Component #4", next2Components, labels);
            next2Components = transformed.Select(
                x => x.Where((y, i) => i < 5 && i >= 3).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #4 vs. Component #5", next2Components, labels);
            next2Components = transformed.Select(
                x => x.Where((y, i) => i < 6 && i >= 4).ToArray()
            ).ToArray();
            ScatterplotBox.Show("Component #5 vs. Component #6", next2Components, labels);

            double[] explainedVariance = pca.Components
                .Select(x => x.CumulativeProportion)
                .Where(x => x < 1)
                .ToArray();

            DataSeriesBox.Show(
                explainedVariance.Select((x, i) => (double)i),
                explainedVariance
            ).SetTitle("Explained Variance");
            System.IO.File.WriteAllLines(
                Path.Combine(dataDirPath, "explained-variance.csv"),
                explainedVariance.Select((x, i) => String.Format("{0},{1:0.0000}", i, x))
            );

            Console.WriteLine("* Exporting pca-transformed feature set...");
            System.IO.File.WriteAllLines(
                Path.Combine(
                    dataDirPath,
                    "pca-transformed-features.csv"
                ),
                transformed.Select(x => String.Join(",", x))
            );
            System.IO.File.WriteAllLines(
                Path.Combine(
                    dataDirPath,
                    "pca-transformed-labels.csv"
                ),
                labels.Select(x => x.ToString())
            );


            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }

        private static double[][] BuildJaggedArray(double[,] ary2d, int rowCount, int colCount)
        {
            double[][] matrix = new double[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                matrix[i] = new double[colCount];
                for (int j = 0; j < colCount; j++)
                {
                    matrix[i][j] = double.IsNaN(ary2d[i, j]) ? 0.0 : ary2d[i, j];
                }
            }
            return matrix;
        }

        private static Frame<int, string> EncodeOneHot(Series<int, string> rows, string originalColName)
        {

            var categoriesByRows = rows.GetAllValues().Select((x, i) =>
            {
                // Encode the categories appeared in each row with 1
                var sb = new SeriesBuilder<string, int>();
                sb.Add(String.Format("{0}_{1}", originalColName, x.Value), 1);

                return KeyValue.Create(i, sb.Series);
            });

            // Create a data frame from the rows we just created
            // And encode missing values with 0
            var categoriesDF = Frame.FromRows(categoriesByRows).FillMissing(0);

            return categoriesDF;
        }
    }
}
