using Accord.Controls;
using Accord.Statistics.Analysis;
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

            // Read in the Cyber Attack dataset
            // TODO: change the path to point to your data directory
            string dataDirPath = @"\\Mac\Home\Documents\c-sharp-machine-learning\ch.9\input-data";

            // Load the data into a data frame
            string dataPath = Path.Combine(dataDirPath, "kddcup.data_10_percent");
            Console.WriteLine("Loading {0}\n\n", dataPath);
            var featuresDF = Frame.ReadCsv(
                dataPath,
                hasHeaders: false,
                inferTypes: true
            );

            string[] colnames =
            {
                "duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                "num_failed_logins", "logged_in", "num_compromised", "root_shell",
                "su_attempted", "num_root", "num_file_creations", "num_shells",
                "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
                "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
                "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
                "attack_type"
            };
            featuresDF.RenameColumns(colnames);

            Console.WriteLine("* Shape: {0}, {1}\n\n", featuresDF.RowCount, featuresDF.ColumnCount);

            // keeping "normal" for now for plotting purposes
            IDictionary<string, string> attackCategories = new Dictionary<string, string>
            {
                {"back", "dos"},
                {"land", "dos"},
                {"neptune", "dos"},
                {"pod", "dos"},
                {"smurf", "dos"},
                {"teardrop", "dos"},
                {"ipsweep", "probe"},
                {"nmap", "probe"},
                {"portsweep", "probe"},
                {"satan", "probe"},
                {"ftp_write", "r2l"},
                {"guess_passwd", "r2l"},
                {"imap", "r2l"},
                {"multihop", "r2l"},
                {"phf", "r2l"},
                {"spy", "r2l"},
                {"warezclient", "r2l"},
                {"warezmaster", "r2l"},
                {"buffer_overflow", "u2r"},
                {"loadmodule", "u2r"},
                {"perl", "u2r"},
                {"rootkit", "u2r"},
                {"normal", "normal"}
            };

            featuresDF.AddColumn(
                "attack_category",
                featuresDF.GetColumn<string>("attack_type")
                    .Select(x => attackCategories[x.Value.Replace(".", "")])
            );

            // Export with Categories
            Console.WriteLine("* Exporting data...");
            featuresDF.SaveCsv(Path.Combine(dataDirPath, "data.csv"));

            // 1. Target Variable Distribution
            Console.WriteLine("\n\n-- Counts by Attack Category --\n");
            var attackCount = featuresDF.AggregateRowsBy<string, int>(
                new string[] { "attack_category" },
                new string[] { "duration" },
                x => x.ValueCount
            ).SortRows("duration");
            attackCount.RenameColumns(new string[] { "attack_category", "count" });

            attackCount.Print();

            DataBarBox.Show(
                attackCount.GetColumn<string>("attack_category").Values.ToArray(),
                attackCount["count"].Values.ToArray()
            ).SetTitle(
                "Counts by Attack Category"
            );

            // Now, remove normal records
            var attackSubset = featuresDF.Rows[
                featuresDF.GetColumn<string>("attack_category").Where(
                    x => !x.Value.Equals("normal")
                ).Keys
            ];
            var normalSubset = featuresDF.Rows[
                featuresDF.GetColumn<string>("attack_category").Where(
                    x => x.Value.Equals("normal")
                ).Keys
            ];

            // 2. Categorical Variable Distribution
            string[] categoricalVars =
            {
                "protocol_type", "service", "flag", "land"
            };
            foreach (string variable in categoricalVars)
            {
                Console.WriteLine("\n\n-- Counts by {0} --\n", variable);
                Console.WriteLine("* Attack:");
                var attackCountDF = attackSubset.AggregateRowsBy<string, int>(
                    new string[] { variable },
                    new string[] { "duration" },
                    x => x.ValueCount
                );
                attackCountDF.RenameColumns(new string[] { variable, "count" });

                attackCountDF.SortRows("count").Print();

                Console.WriteLine("* Normal:");
                var countDF = normalSubset.AggregateRowsBy<string, int>(
                    new string[] { variable },
                    new string[] { "duration" },
                    x => x.ValueCount
                );
                countDF.RenameColumns(new string[] { variable, "count" });

                countDF.SortRows("count").Print();

                DataBarBox.Show(
                    countDF.GetColumn<string>(variable).Values.ToArray(),
                    new double[][] 
                    {
                        attackCountDF["count"].Values.ToArray(),
                        countDF["count"].Values.ToArray()
                    }
                ).SetTitle(
                    String.Format("Counts by {0} (0 - Attack, 1 - Normal)", variable)
                );
            }

            // 3. Continuous Variable Distribution
            string[] continuousVars =
            {
                "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
                "num_failed_logins", "num_compromised", "root_shell", "su_attempted",
                "num_root", "num_file_creations", "num_shells", "num_access_files",
                "num_outbound_cmds", "count", "srv_count", "serror_rate", "srv_serror_rate",
                "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
            };

            foreach (string variable in continuousVars)
            {
                Console.WriteLine(String.Format("\n\n-- {0} Distribution (Attack) -- ", variable));
                double[] attachQuartiles = Accord.Statistics.Measures.Quantiles(
                    attackSubset[variable].DropMissing().ValuesAll.ToArray(),
                    new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
                );
                Console.WriteLine(
                    "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                    attachQuartiles[0], attachQuartiles[1], attachQuartiles[2], attachQuartiles[3], attachQuartiles[4]
                );

                Console.WriteLine(String.Format("\n\n-- {0} Distribution (Normal) -- ", variable));
                double[] normalQuantiles = Accord.Statistics.Measures.Quantiles(
                    normalSubset[variable].DropMissing().ValuesAll.ToArray(),
                    new double[] { 0, 0.25, 0.5, 0.75, 1.0 }
                );
                Console.WriteLine(
                    "Min: \t\t\t{0:0.00}\nQ1 (25% Percentile): \t{1:0.00}\nQ2 (Median): \t\t{2:0.00}\nQ3 (75% Percentile): \t{3:0.00}\nMax: \t\t\t{4:0.00}",
                    normalQuantiles[0], normalQuantiles[1], normalQuantiles[2], normalQuantiles[3], normalQuantiles[4]
                );
            }


            Console.WriteLine("\n\n\n\n\nDONE!!!");
            Console.ReadKey();
        }
    }
}
