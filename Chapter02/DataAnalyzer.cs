using Accord.Controls;
using Deedle;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;


namespace DataAnalyzer
{
    class Program
    {
        private static Frame<int, string> CreateWordVec(Series<int, string> rows)
        {
            var wordsByRows = rows.GetAllValues().Select((x, i) =>
            {
                var sb = new SeriesBuilder<string, int>();

                ISet<string> words = new HashSet<string>(
                    Regex.Matches(
                        // Alphabetical characters only
                        x.Value, "[a-zA-Z]+('(s|d|t|ve|m))?"
                    ).Cast<Match>().Select(
                        // Then, convert each word to lowercase
                        y => y.Value.ToLower()
                    ).ToArray()
                );

                // Encode words appeared in each row with 1
                foreach (string w in words)
                {
                    sb.Add(w, 1);
                }

                return KeyValue.Create(i, sb.Series);
            });

            // Create a data frame from the rows we just created
            // And encode missing values with 0
            var wordVecDF = Frame.FromRows(wordsByRows).FillMissing(0);

            return wordVecDF;
        }


        static void Main(string[] args)
        {
            Console.SetWindowSize(125, 50);
            // Read in the file we created in the Data Preparation step
            // TODO: change the path to point to your data directory
            string dataDirPath = "\\\\Mac\\Home\\Documents\\c-sharp-machine-learning\\ch.2\\output";
            // Read in stopwords list
            ISet<string> stopWords = new HashSet<string>(
                File.ReadLines("\\\\Mac\\Home\\Documents\\c-sharp-machine-learning\\ch.2\\stopwords.txt")
            );

            // Load the data into a data frame and set the "emailNum" column as an index
            var rawDF = Frame.ReadCsv(
                Path.Combine(dataDirPath, "data-preparation-step\\transformed.csv"),
                hasHeaders: true,
                inferTypes: false,
                schema: "int,string,string,int"
            ).IndexRows<int>("emailNum").SortRowsByKey(); ;

            // Look at words used in Subject lines
            var subjectWordVecDF = CreateWordVec(rawDF.GetColumn<string>("subject"));
            subjectWordVecDF.SaveCsv(Path.Combine(dataDirPath, "data-preparation-step\\subjectWordVec-alphaonly.csv"));
            Console.WriteLine("* Subject Word Vec DF Shape ({0}, {1})", subjectWordVecDF.RowCount, subjectWordVecDF.ColumnCount);

            // Get term frequencies by each group (ham vs. spam)
            var hamEmailCount = rawDF.GetColumn<int>("is_ham").NumSum();
            var spamEmailCount = subjectWordVecDF.RowCount - hamEmailCount;

            subjectWordVecDF.AddColumn("is_ham", rawDF.GetColumn<int>("is_ham"));
            var hamTermFrequencies = subjectWordVecDF.Where(
                x => x.Value.GetAs<int>("is_ham") == 1
            ).Sum().Sort().Reversed.Where(x => x.Key != "is_ham");

            var spamTermFrequencies = subjectWordVecDF.Where(
                x => x.Value.GetAs<int>("is_ham") == 0
            ).Sum().Sort().Reversed;

            // Look at Top 10 terms that appear in Ham vs. Spam emails
            var topN = 10;

            var hamTermProportions = hamTermFrequencies / hamEmailCount;
            var topHamTerms = hamTermProportions.Keys.Take(topN);
            var topHamTermsProportions = hamTermProportions.Values.Take(topN);

            System.IO.File.WriteAllLines(
                dataDirPath + "\\ham-frequencies.csv",
                hamTermFrequencies.Keys.Zip(
                    hamTermFrequencies.Values, (a, b) => string.Format("{0},{1}", a, b)
                )
            );

            var spamTermProportions = spamTermFrequencies / spamEmailCount;
            var topSpamTerms = spamTermProportions.Keys.Take(topN);
            var topSpamTermsProportions = spamTermProportions.Values.Take(topN);

            System.IO.File.WriteAllLines(
                dataDirPath + "\\spam-frequencies.csv",
                spamTermFrequencies.Keys.Zip(
                    spamTermFrequencies.Values, (a, b) => string.Format("{0},{1}", a, b)
                )
            );

            var barChart = DataBarBox.Show(
                new string[] { "Ham", "Spam" },
                new double[] {
                    hamEmailCount,
                    spamEmailCount
                }
            );
            barChart.SetTitle("Ham vs. Spam in Sample Set");

            var hamBarChart = DataBarBox.Show(
                topHamTerms.ToArray(),
                new double[][] {
                    topHamTermsProportions.ToArray(),
                    spamTermProportions.GetItems(topHamTerms).Values.ToArray()
                }
            );
            hamBarChart.SetTitle("Top 10 Terms in Ham Emails (blue: HAM, red: SPAM)");
            System.Threading.Thread.Sleep(3000);
            hamBarChart.Invoke(
                new Action(() =>
                {
                    hamBarChart.Size = new System.Drawing.Size(5000, 1500);
                })
            );

            var spamBarChart = DataBarBox.Show(
                topSpamTerms.ToArray(),
                new double[][] {
                    hamTermProportions.GetItems(topSpamTerms).Values.ToArray(),
                    topSpamTermsProportions.ToArray()
                }
            );
            spamBarChart.SetTitle("Top 10 Terms in Spam Emails (blue: HAM, red: SPAM)");
            System.Threading.Thread.Sleep(3000);
            spamBarChart.Invoke(
                new Action(() =>
                {
                    spamBarChart.Size = new System.Drawing.Size(5000, 1500);
                })
            );

            // Look at top terms appear in Ham vs. Spam emails after filtering out stopwords
            var hamTermFrequenciesAfterStopWords = hamTermFrequencies.Where(
                x => !stopWords.Contains(x.Key)
            );
            var hamTermProportionsAfterStopWords = hamTermProportions.Where(
                x => !stopWords.Contains(x.Key)
            );
            var topHamTermsAfterStopWords = hamTermProportionsAfterStopWords.Keys.Take(topN);
            var topHamTermsProportionsAfterStopWords = hamTermProportionsAfterStopWords.Values.Take(topN);
            System.IO.File.WriteAllLines(
                dataDirPath + "\\ham-frequencies-after-stopwords.csv",
                hamTermFrequenciesAfterStopWords.Keys.Zip(
                    hamTermFrequenciesAfterStopWords.Values, (a, b) => string.Format("{0},{1}", a, b)
                )
            );

            var spamTermFrequenciesAfterStopWords = spamTermFrequencies.Where(
                x => !stopWords.Contains(x.Key)
            );
            var spamTermProportionsAfterStopWords = spamTermProportions.Where(
                x => !stopWords.Contains(x.Key)
            );
            var topSpamTermsAfterStopWords = spamTermProportionsAfterStopWords.Keys.Take(topN);
            var topSpamTermsProportionsAfterStopWords = spamTermProportionsAfterStopWords.Values.Take(topN);
            System.IO.File.WriteAllLines(
                dataDirPath + "\\spam-frequencies-after-stopwords.csv",
                spamTermFrequenciesAfterStopWords.Keys.Zip(
                    spamTermFrequenciesAfterStopWords.Values, (a, b) => string.Format("{0},{1}", a, b)
                )
            );

            hamBarChart = DataBarBox.Show(
                topHamTermsAfterStopWords.ToArray(),
                new double[][] {
                    topHamTermsProportionsAfterStopWords.ToArray(),
                    spamTermProportionsAfterStopWords.GetItems(topHamTermsAfterStopWords).Values.ToArray()
                }
            );
            hamBarChart.SetTitle("Top 10 Terms in Ham Emails - after filtering out stopwords (blue: HAM, red: SPAM)");
            System.Threading.Thread.Sleep(3000);
            hamBarChart.Invoke(
                new Action(() =>
                {
                    hamBarChart.Size = new System.Drawing.Size(5000, 1500);
                })
            );

            spamBarChart = DataBarBox.Show(
                topSpamTermsAfterStopWords.ToArray(),
                new double[][] {
                    hamTermProportionsAfterStopWords.GetItems(topSpamTermsAfterStopWords).Values.ToArray(),
                    topSpamTermsProportionsAfterStopWords.ToArray()
                }
            );
            spamBarChart.SetTitle("Top 10 Terms in Spam Emails - after filtering out stopwords (blue: HAM, red: SPAM)");
            System.Threading.Thread.Sleep(3000);
            spamBarChart.Invoke(
                new Action(() =>
                {
                    spamBarChart.Size = new System.Drawing.Size(5000, 1500);
                })
            );

            Console.WriteLine("Data Analysis Step Done!");
            Console.ReadKey();
        }
    }
}
