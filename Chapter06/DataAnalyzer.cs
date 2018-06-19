using Accord.Controls;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataAnalyzer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.SetWindowSize(100, 60);

            // Read in the Online Retail dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.6\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "data.csv");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var ecommerceDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: true,
                inferTypes: true
            );

            Console.WriteLine("* Shape: {0}, {1}\n\n", ecommerceDF.RowCount, ecommerceDF.ColumnCount);

            // 1. Missing CustomerID Values
            ecommerceDF
                .Columns[new string[] { "CustomerID", "InvoiceNo", "StockCode", "Quantity", "UnitPrice", "Country" }]
                .GetRowsAt(new int[] { 1440, 1441, 1442, 1443, 1444, 1445, 1446 })
                .Print();
            Console.WriteLine("\n\n* # of values in CustomerID column: {0}", ecommerceDF["CustomerID"].ValueCount);
            // Drop missing values
            ecommerceDF = ecommerceDF
                .Columns[new string[] { "CustomerID", "Description", "Quantity", "UnitPrice", "Country" }]
                .DropSparseRows();
            // Per-Transaction Purchase Amount = Quantity * UnitPrice
            ecommerceDF.AddColumn("Amount", ecommerceDF["Quantity"] * ecommerceDF["UnitPrice"]);

            Console.WriteLine("\n\n* Shape (After dropping missing values): {0}, {1}\n", ecommerceDF.RowCount, ecommerceDF.ColumnCount);
            Console.WriteLine("* After dropping missing values and unnecessary columns:");
            ecommerceDF.GetRowsAt(new int[] { 0, 1, 2, 3, 4 }).Print();
            // Export Data
            ecommerceDF.SaveCsv(Path.Combine(dataDirPath, "data-clean.csv"));

            // 2. Number of transactions by country
            var numTransactionsByCountry = ecommerceDF
                .AggregateRowsBy<string, int>(
                    new string[] { "Country" },
                    new string[] { "CustomerID" },
                    x => x.ValueCount
                ).SortRows("CustomerID");

            var top5 = numTransactionsByCountry
                .GetRowsAt(new int[] {
                    numTransactionsByCountry.RowCount-1,  numTransactionsByCountry.RowCount-2,
                    numTransactionsByCountry.RowCount-3, numTransactionsByCountry.RowCount-4,
                    numTransactionsByCountry.RowCount-5 });
            top5.Print();

            var topTransactionByCountryBarChart = DataBarBox.Show(
                top5.GetColumn<string>("Country").Values.ToArray().Select(x => x.Equals("United Kingdom") ? "UK" : x),
                top5["CustomerID"].Values.ToArray()
            );
            topTransactionByCountryBarChart.SetTitle(
                "Top 5 Countries with the most number of transactions"
             );

            // 3. Per-Transaction Quantity Distributions
            Console.WriteLine("\n\n-- Per-Transaction Order Quantity Distribution-- ");
            double[] quantiles = Accord.Statistics.Measures.Quantiles(
                ecommerceDF["Quantity"].ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );

            Console.WriteLine("\n\n-- Per-Transaction Purchase-Order Quantity Distribution-- ");
            quantiles = Accord.Statistics.Measures.Quantiles(
                ecommerceDF["Quantity"].Where(x => x.Value >= 0).ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );

            Console.WriteLine("\n\n-- Per-Transaction Cancel-Order Quantity Distribution-- ");
            quantiles = Accord.Statistics.Measures.Quantiles(
                ecommerceDF["Quantity"].Where(x => x.Value < 0).ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );

            // 4. Per-Transaction Unit Price Distributions
            Console.WriteLine("\n\n-- Per-Transaction Unit Price Distribution-- ");
            quantiles = Accord.Statistics.Measures.Quantiles(
                ecommerceDF["UnitPrice"].ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );

            // 5. Per-Transaction Purchase Price Distributions
            Console.WriteLine("\n\n-- Per-Transaction Total Amount Distribution-- ");
            quantiles = Accord.Statistics.Measures.Quantiles(
                ecommerceDF["Amount"].ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );

            Console.WriteLine("\n\n-- Per-Transaction Purchase-Order Total Amount Distribution-- ");
            quantiles = Accord.Statistics.Measures.Quantiles(
                ecommerceDF["Amount"].Where(x => x.Value >= 0).ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );

            Console.WriteLine("\n\n-- Per-Transaction Cancel-Order Total Amount Distribution-- ");
            quantiles = Accord.Statistics.Measures.Quantiles(
                ecommerceDF["Amount"].Where(x => x.Value < 0).ValuesAll.ToArray(),
                new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
            );
            Console.WriteLine(
                "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4]
            );

            // 6. # of Purchase vs. Cancelled Transactions
            var purchaseVSCancelBarChart = DataBarBox.Show(
                new string[] { "Purchase", "Cancel" },
                new double[] {
                    ecommerceDF["Quantity"].Where(x => x.Value >= 0).ValueCount ,
                    ecommerceDF["Quantity"].Where(x => x.Value < 0).ValueCount
                }
            );
            purchaseVSCancelBarChart.SetTitle(
                "Purchase vs. Cancel"
             );


            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }
    }
}
