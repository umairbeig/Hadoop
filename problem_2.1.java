package org.example;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Set;
import java.util.HashSet;
import java.util.stream.Stream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import opennlp.tools.stemmer.PorterStemmer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
public class DocumentFrequency {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text>
    {
        private Text word = new Text();
        private Text file = new Text();

        private Set<String> stopWords = new HashSet<String>();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            Path[] stopWordsFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            for (Path stopWordsFile : stopWordsFiles) {
                if (stopWordsFile.getName().equals("stopwords.txt")) {
                    BufferedReader reader = new BufferedReader(new FileReader(stopWordsFile.toString()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        stopWords.add(line.toLowerCase());
                    }
                    reader.close();
                }
            }
        }

        @Override
        public void map(Object Key, Text Value, Context context) throws IOException, InterruptedException
        {
            String line = Value.toString();

            // Split lines into words.
            String[] tokens = line.split("[^\\w']+");

            // initialising the stemmer to stem the words in the document.
            PorterStemmer stemmer = new PorterStemmer();
            String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();




            for(String token : tokens)
            {
                token=token.toLowerCase();
//                System.out.println(stopWords);
                if (stopWords.contains(token)) {
//                    System.out.println("stopword found");
                    continue;
                }

                String stem_token = stemmer.stem(token).toString();
                word.set(stem_token);
                file.set(fileName);
                context.write(word, file);
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, Text, Text, IntWritable> {
        // stores the value of the doc frequency for the token.
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Keeps a track of all unique words encountered till now.
            Set<Text> unique = new HashSet<Text>();

            for (Text val : values) {
                unique.add(val);
            }
            result.set(unique.size());
            context.write(key, result);
        }
    }
    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "DocumentFrequency");

        // setting the Mapper class for the job
        job.setJarByClass(DocumentFrequency.class);
        job.setMapperClass(TokenizerMapper.class);
        // setting the Reducer class for the job
        job.setReducerClass(IntSumReducer.class);
        // Setting the datatype for the map output key, value pair
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        // The output file datatype for key value
        // are Text and IntWritable
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        // setting input path using arguments given while execution
        // Format: hadoop jar <jar path> className input file output file
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        DistributedCache.addCacheFile(new Path(args[2]).toUri(),job.getConfiguration());

        // The program will run till the job is completed.
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

}

