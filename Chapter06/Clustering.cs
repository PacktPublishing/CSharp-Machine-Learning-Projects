using Accord.Controls;
using Accord.MachineLearning;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Clustering
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 50);

            // Read in the Online Retail feature dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.6\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "features.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var ecommerceDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );
            Console.WriteLine("* Shape: {0}, {1}", ecommerceDF.RowCount, ecommerceDF.ColumnCount);

            string[] features = new string[] { "NetRevenuePercentile", "AvgUnitPricePercentile", "AvgQuantityPercentile" };
            Console.WriteLine("* Features: {0}\n\n", String.Join(", ", features));

            var normalizedDf = Frame.CreateEmpty<int, string>();
            var average = ecommerceDF.Columns[features].Sum() / ecommerceDF.RowCount;
            foreach(string feature in features)
            {
                normalizedDf.AddColumn(feature, (ecommerceDF[feature] - average[feature]) / ecommerceDF[feature].StdDev());
            }

            double[][] sampleSet = BuildJaggedArray(
                normalizedDf.Columns[features].ToArray2D<double>(),
                normalizedDf.RowCount,
                features.Length
            );

            // Create a new K-Means algorithm with n clusters
            Accord.Math.Random.Generator.Seed = 0;

            int[] numClusters = new int[] { 4, 5, 6, 7, 8 };
            List<string> clusterNames = new List<string>();
            List<double> silhouetteScores = new List<double>();
            for(int i = 0; i < numClusters.Length; i++)
            {
                KMeans kmeans = new KMeans(numClusters[i]);
                KMeansClusterCollection clusters = kmeans.Learn(sampleSet);
                int[] labels = clusters.Decide(sampleSet);

                string colname = String.Format("Cluster-{0}", numClusters[i]);
                clusterNames.Add(colname);

                normalizedDf.AddColumn(colname, labels);
                ecommerceDF.AddColumn(colname, labels);

                Console.WriteLine("\n\n\n#####################    {0}    ###########################", colname);

                Console.WriteLine("\n\n* Centroids for {0} clusters:", numClusters[i]);

                PrintCentroidsInfo(clusters.Centroids, features);
                Console.WriteLine("\n");

                VisualizeClusters(normalizedDf, colname, "NetRevenuePercentile", "AvgUnitPricePercentile");
                VisualizeClusters(normalizedDf, colname, "AvgUnitPricePercentile", "AvgQuantityPercentile");
                VisualizeClusters(normalizedDf, colname, "NetRevenuePercentile", "AvgQuantityPercentile");

                for (int j = 0; j < numClusters[i]; j++)
                {
                    GetTopNItemsPerCluster(ecommerceDF, j, colname);
                }

                double silhouetteScore = CalculateSilhouetteScore(normalizedDf, features, numClusters[i], colname);
                Console.WriteLine("\n\n* Silhouette Score: {0}", silhouetteScore.ToString("0.0000"));

                silhouetteScores.Add(silhouetteScore);
                Console.WriteLine("\n\n##############################################################\n\n\n");
            }
            
            for(int i = 0; i < clusterNames.Count; i++)
            {
                Console.WriteLine("- Silhouette Score for {0}: {1}", clusterNames[i], silhouetteScores[i].ToString("0.0000"));
            }

            Console.WriteLine("\n\n\nDONE!!");
            Console.ReadKey();
        }

        private static double CalculateSilhouetteScore(Frame<int, string> df, string[] features, int numCluster, string clusterColname)
        {
            double[][] data = BuildJaggedArray(df.Columns[features].ToArray2D<double>(), df.RowCount, features.Length);

            double total = 0.0;
            for(int i = 0; i < df.RowCount; i++)
            {
                double sameClusterAverageDistance = 0.0;
                double differentClusterDistance = 1000000.0;

                double[] point = df.Columns[features].GetRowAt<double>(i).Values.ToArray();
                double cluster = df[clusterColname].GetAt(i);

                for(int j = 0; j < numCluster; j++)
                {
                    double averageDistance = CalculateAverageDistance(df, features, clusterColname, j, point);

                    if (cluster == j)
                    {
                        sameClusterAverageDistance = averageDistance;
                    } else
                    {
                        differentClusterDistance = Math.Min(averageDistance, differentClusterDistance);
                    }
                }

                total += (differentClusterDistance - sameClusterAverageDistance) / Math.Max(sameClusterAverageDistance, differentClusterDistance);
            }

            return total / df.RowCount;
        }

        private static double CalculateAverageDistance(Frame<int, string> df, string[] features, string clusterColname, int cluster, double[] point)
        {
            var clusterDF = df.Rows[
                df[clusterColname].Where(x => (int)x.Value == cluster).Keys
            ];
            double[][] clusterData = BuildJaggedArray(
                clusterDF.Columns[features].ToArray2D<double>(),
                clusterDF.RowCount,
                features.Length
            );

            double averageDistance = 0.0;
            for (int i = 0; i < clusterData.Length; i++)
            {
                averageDistance += Math.Sqrt(
                    point.Select((x, j) => Math.Pow(x - clusterData[i][j], 2)).Sum()
                );
            }
            averageDistance /= (float)clusterData.Length;

            return averageDistance;
        }

        private static void GetTopNItemsPerCluster(Frame<int, string> ecommerceDF, int cluster, string colname)
        {
            var itemDistribution = ecommerceDF.Rows[
                ecommerceDF.GetColumn<int>(colname).Where(x => x.Value == cluster).Keys
            ].GetColumn<string>("Description")
            .GroupBy<string>(x => x.Value)
            .Select(x => (double)x.Value.KeyCount)
            .Sort().Reversed;

            Console.WriteLine("\n# {0} of {1} - Top 10:", cluster+1, colname);
            itemDistribution[itemDistribution.Keys.Take(10)].Print();
        }

        private static void PrintCentroidsInfo(double[][] centroids, string[] features)
        {
            Console.WriteLine("\t{0}", String.Join("\t", features));
            for (int i = 0; i < centroids.Length; i++)
            {
                Console.Write("{0}:\t", i);
                Console.WriteLine(String.Join("\t", centroids[i]));
            }
        }

        private static void VisualizeClusters(Frame<int, string> df, string colname, string feature1, string feature2)
        {
            var scatterplot = ScatterplotBox.Show(
                String.Format("{0} vs. {1}", feature1, feature2),
                BuildJaggedArray(
                    df.Columns[new string[] { feature1, feature2 }].ToArray2D<double>(),
                    df.RowCount,
                    2
                ),
                df.GetColumn<int>(colname).Values.ToArray()
            ).SetTitle(colname);
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
