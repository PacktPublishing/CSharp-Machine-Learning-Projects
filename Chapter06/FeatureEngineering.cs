using Accord.Controls;
using CenterSpace.NMath.Stats;
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
            Console.SetWindowSize(100, 50);

            // Read in the Online Retail dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.6\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "data-clean.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var ecommerceDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );
            Console.WriteLine("* Shape: {0}, {1}\n\n", ecommerceDF.RowCount, ecommerceDF.ColumnCount);

            // 1. Net Revenue per Customer
            var revPerCustomerDF = ecommerceDF.AggregateRowsBy<double, double>(
                new string[] { "CustomerID" },
                new string[] { "Amount" },
                x => x.Sum()
            );
            // 2. # of Total Transactions per Customer
            var numTransactionsPerCustomerDF = ecommerceDF.AggregateRowsBy<double, double>(
                new string[] { "CustomerID" },
                new string[] { "Quantity" },
                x => x.ValueCount
            );
            // 3. # of Cancelled Transactions per Customer
            var numCancelledPerCustomerDF = ecommerceDF.AggregateRowsBy<double, double>(
                new string[] { "CustomerID" },
                new string[] { "Quantity" },
                x => x.Select(y => y.Value >= 0 ? 0.0 : 1.0).Sum()
            );
            // 4. Average UnitPrice per Customer
            var avgUnitPricePerCustomerDF = ecommerceDF.AggregateRowsBy<double, double>(
                new string[] { "CustomerID" },
                new string[] { "UnitPrice" },
                x => x.Sum() / x.ValueCount
            );
            // 5. Average Quantity per Customer
            var avgQuantityPerCustomerDF = ecommerceDF.AggregateRowsBy<double, double>(
                new string[] { "CustomerID" },
                new string[] { "Quantity" },
                x => x.Sum() / x.ValueCount
            );

            // Aggregate all results
            var featuresDF = Frame.CreateEmpty<int, string>();
            featuresDF.AddColumn("CustomerID", revPerCustomerDF.GetColumn<double>("CustomerID"));
            featuresDF.AddColumn("Description", ecommerceDF.GetColumn<string>("Description"));
            featuresDF.AddColumn("NetRevenue", revPerCustomerDF.GetColumn<double>("Amount"));
            featuresDF.AddColumn("NumTransactions", numTransactionsPerCustomerDF.GetColumn<double>("Quantity"));
            featuresDF.AddColumn("NumCancelled", numCancelledPerCustomerDF.GetColumn<double>("Quantity"));
            featuresDF.AddColumn("AvgUnitPrice", avgUnitPricePerCustomerDF.GetColumn<double>("UnitPrice"));
            featuresDF.AddColumn("AvgQuantity", avgQuantityPerCustomerDF.GetColumn<double>("Quantity"));
            featuresDF.AddColumn("PercentageCancelled", featuresDF["NumCancelled"] / featuresDF["NumTransactions"]);

            Console.WriteLine("\n\n* Feature Set:");
            featuresDF.Print();

            // NetRevenue feature distribution
            PrintQuartiles(featuresDF, "NetRevenue");
            // NumTransactions feature distribution
            PrintQuartiles(featuresDF, "NumTransactions");
            // AvgUnitPrice feature distribution
            PrintQuartiles(featuresDF, "AvgUnitPrice");
            // AvgQuantity feature distribution
            PrintQuartiles(featuresDF, "AvgQuantity");
            // PercentageCancelled feature distribution
            PrintQuartiles(featuresDF, "PercentageCancelled");
            Console.WriteLine("\n\n* Feature DF Shape: ({0}, {1})", featuresDF.RowCount, featuresDF.ColumnCount);

            // 1. Drop Customers with Negative NetRevenue
            featuresDF = featuresDF.Rows[
                featuresDF["NetRevenue"].Where(x => x.Value >= 0.0).Keys
            ];
            // 2. Drop Customers with Negative AvgQuantity
            featuresDF = featuresDF.Rows[
                featuresDF["AvgQuantity"].Where(x => x.Value >= 0.0).Keys
            ];
            // 3. Drop Customers who have more cancel orders than purchase orders
            featuresDF = featuresDF.Rows[
                featuresDF["PercentageCancelled"].Where(x => x.Value < 0.5).Keys
            ];

            Console.WriteLine("\n\n\n\n* After dropping customers with potential orphan cancel orders:");
            // NetRevenue feature distribution
            PrintQuartiles(featuresDF, "NetRevenue");
            // NumTransactions feature distribution
            PrintQuartiles(featuresDF, "NumTransactions");
            // AvgUnitPrice feature distribution
            PrintQuartiles(featuresDF, "AvgUnitPrice");
            // AvgQuantity feature distribution
            PrintQuartiles(featuresDF, "AvgQuantity");
            // PercentageCancelled feature distribution
            PrintQuartiles(featuresDF, "PercentageCancelled");
            Console.WriteLine("\n\n* Feature DF Shape: ({0}, {1})", featuresDF.RowCount, featuresDF.ColumnCount);

            HistogramBox.CheckForIllegalCrossThreadCalls = false;
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["NetRevenue"].ValuesAll.ToArray(),
                title: "NetRevenue Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["NumTransactions"].ValuesAll.ToArray(),
                title: "NumTransactions Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["AvgUnitPrice"].ValuesAll.ToArray(),
                title: "AvgUnitPrice Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["AvgQuantity"].ValuesAll.ToArray(),
                title: "AvgQuantity Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["PercentageCancelled"].ValuesAll.ToArray(),
                title: "PercentageCancelled Distribution"
            )
            .SetNumberOfBins(50);


            // Create Percentile Features
            featuresDF.AddColumn(
                "NetRevenuePercentile",
                featuresDF["NetRevenue"].Select(
                    x => StatsFunctions.PercentileRank(featuresDF["NetRevenue"].Values.ToArray(), x.Value)
                )
            );
            featuresDF.AddColumn(
                "NumTransactionsPercentile",
                featuresDF["NumTransactions"].Select(
                    x => StatsFunctions.PercentileRank(featuresDF["NumTransactions"].Values.ToArray(), x.Value)
                )
            );
            featuresDF.AddColumn(
                "AvgUnitPricePercentile",
                featuresDF["AvgUnitPrice"].Select(
                    x => StatsFunctions.PercentileRank(featuresDF["AvgUnitPrice"].Values.ToArray(), x.Value)
                )
            );
            featuresDF.AddColumn(
                "AvgQuantityPercentile",
                featuresDF["AvgQuantity"].Select(
                    x => StatsFunctions.PercentileRank(featuresDF["AvgQuantity"].Values.ToArray(), x.Value)
                )
            );
            featuresDF.AddColumn(
                "PercentageCancelledPercentile",
                featuresDF["PercentageCancelled"].Select(
                    x => StatsFunctions.PercentileRank(featuresDF["PercentageCancelled"].Values.ToArray(), x.Value)
                )
            );
            Console.WriteLine("\n\n\n* Percentile Features:");
            featuresDF.Columns[
                new string[] { "NetRevenue", "NetRevenuePercentile", "NumTransactions", "NumTransactionsPercentile" }
            ].Print();

            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["NetRevenuePercentile"].ValuesAll.ToArray(),
                title: "NetRevenuePercentile Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["NumTransactionsPercentile"].ValuesAll.ToArray(),
                title: "NumTransactionsPercentile Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["AvgUnitPricePercentile"].ValuesAll.ToArray(),
                title: "AvgUnitPricePercentile Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["AvgQuantityPercentile"].ValuesAll.ToArray(),
                title: "AvgQuantityPercentile Distribution"
            )
            .SetNumberOfBins(50);
            HistogramBox
            .Show(
                featuresDF.DropSparseRows()["PercentageCancelledPercentile"].ValuesAll.ToArray(),
                title: "PercentageCancelledPercentile Distribution"
            )
            .SetNumberOfBins(50);

            string outputPath = Path.Combine(dataDirPath, "features.csv");
            Console.WriteLine("* Exporting features data: {0}", outputPath);
            featuresDF.SaveCsv(outputPath);

            Console.WriteLine("\n\n\n\nDONE!!");
            Console.ReadKey();
        }

        private static void PrintQuartiles(Frame<int, string> df, string colname)
        {
            Console.WriteLine("\n\n-- {0} Distribution-- ", colname);
            double[] quantiles = Accord.Statistics.Measures.Quantiles(
                df[colname].ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );
        }
    }
}
