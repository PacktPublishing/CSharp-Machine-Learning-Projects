using Accord.Controls;
using Deedle;
using edu.stanford.nlp.ling;
using edu.stanford.nlp.pipeline;
using java.util;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TwitterTokenizer
{
    class Program
    {
        private static Frame<int, string> CreateWordVec(Series<int, string> rows, ISet<string> stopWords, bool useLemma=false)
        {
            // Path to the folder with models extracted from `stanford-corenlp-<version>-models.jar`
            var jarRoot = @"<path-to-model-files-dir>";

            // Annotation pipeline configuration
            var props = new Properties();
            props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
            props.setProperty("ner.useSUTime", "0");

            // We should change current directory, so StanfordCoreNLP could find all the model files automatically
            var curDir = Environment.CurrentDirectory;
            Directory.SetCurrentDirectory(jarRoot);
            var pipeline = new StanfordCoreNLP(props);
            Directory.SetCurrentDirectory(curDir);

            var wordsByRows = rows.GetAllValues().Select((x, i) =>
            {
                var sb = new SeriesBuilder<string, int>();

                // Annotation
                var annotation = new Annotation(x.Value);
                pipeline.annotate(annotation);

                var tokens = annotation.get(typeof(CoreAnnotations.TokensAnnotation));
                ISet<string> terms = new HashSet<string>();

                foreach (CoreLabel token in tokens as ArrayList)
                {
                    string lemma = token.lemma().ToLower();
                    string word = token.word().ToLower();
                    string tag = token.tag();
                    //Console.WriteLine("lemma: {0}, word: {1}, tag: {2}", lemma, word, tag);

                    // Filter out stop words and single-charater words
                    if (!stopWords.Contains(lemma) && word.Length > 1)
                    {
                        if (!useLemma)
                        {
                            terms.Add(word);
                        }
                        else
                        {
                            terms.Add(lemma);
                        }
                    }
                }

                foreach (string term in terms)
                {
                    sb.Add(term, 1);
                }

                return KeyValue.Create(i, sb.Series);
            });

            // Create a data frame from the rows we just created
            // And encode missing values with 0
            var wordVecDF = Frame.FromRows(wordsByRows).FillMissing(0);

            return wordVecDF;
        }

        private static void WriteDataFrameRowByRow(Frame<int, string> tweetWordVecDF, string filePath)
        {
            string[] columns = tweetWordVecDF.ColumnKeys.ToArray();

            using (StreamWriter w = File.AppendText(filePath))
            {
                List<string> headers = new List<string>();
                foreach(string col in columns)
                {
                    headers.Add(col.Replace("\"", "\"\""));
                }
                w.WriteLine("\"" + String.Join("\",\"", headers) + "\"");

                for(int i = 0; i < tweetWordVecDF.RowCount; i++)
                {
                    w.WriteLine(String.Join(",", tweetWordVecDF.GetRowAt<int>(i).Values));
                }
            }
        }

        static void Main(string[] args)
        {
            Console.SetWindowSize(150, 80);

            // Read in the file we created in the Data Preparation step
            // TODO: change the path to point to your data directory
            string dataDirPath = @"<path-to-data-dir>";

            // Read in stopwords list that we used in Chapter #2
            ISet<string> stopWords = new HashSet<string>(
                File.ReadLines("<path-to-stopwords.txt>")
            );

            // Load the data into a data frame
            string trainDataPath = Path.Combine(dataDirPath, "processed-training.csv");
            Console.WriteLine("- Loading {0}", trainDataPath);
            var rawDF = Frame.ReadCsv(
                trainDataPath,
                hasHeaders: true,
                inferTypes: true
            );

            // Look at the sentiment distributions in our sample set
            var sampleSetDistribution = rawDF.GetColumn<string>(
                "airline_sentiment"
            ).GroupBy<string>(x => x.Value).Select(x => x.Value.KeyCount);
            sampleSetDistribution.Print();
            Console.WriteLine(String.Join(",", sampleSetDistribution.Values.ToArray()));

            var barChart = DataBarBox.Show(
                new string[] { "neutral", "positive", "negative" },
                sampleSetDistribution.Values.Select(i => (double)i).ToArray()
            );
            barChart.SetTitle("Sentiment Distribution in Sample Set");

            // Look at words in pre-processed Tweets
            var tweetWordVecDF = CreateWordVec(rawDF.GetColumn<string>("tweet"), stopWords, useLemma: false);
            tweetWordVecDF.AddColumn(
                "tweet_polarity",
                rawDF.GetColumn<string>("airline_sentiment").Select(
                    x => x.Value == "neutral" ? 0 : x.Value == "positive" ? 1 : 2
                )
            );
            WriteDataFrameRowByRow(tweetWordVecDF, Path.Combine(dataDirPath, "tweet-words.csv"));
            Console.WriteLine("* Tweet Word Vec DF Shape ({0}, {1})", tweetWordVecDF.RowCount, tweetWordVecDF.ColumnCount);


            // Look at lemmas in pre-processed Tweets
            var tweetLemmaVecDF = CreateWordVec(rawDF.GetColumn<string>("tweet"), stopWords, useLemma: true);
            tweetLemmaVecDF.AddColumn(
                "tweet_polarity", 
                rawDF.GetColumn<string>("airline_sentiment").Select(
                    x => x.Value == "neutral" ? 0 : x.Value == "positive" ? 1 : 2
                )
            );
            WriteDataFrameRowByRow(tweetLemmaVecDF, Path.Combine(dataDirPath, "tweet-lemma.csv"));
            Console.WriteLine("* Tweet Lemma Vec DF Shape ({0}, {1})", tweetLemmaVecDF.RowCount, tweetLemmaVecDF.ColumnCount);

            Console.WriteLine("Done!!!");
            Console.ReadKey();
        }
    }
}
