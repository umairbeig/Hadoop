package org.example;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.SimpleTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

public class PosTaggingPairs {

    public static class POSMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private SimpleTokenizer tokenizer;
        private POSTaggerME tagger;


        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            try {
                InputStream modelIn = getClass().getResourceAsStream("/model.bin");

                if (modelIn == null) {
                    throw new FileNotFoundException("Model file not found.");
                }

                // Load the POS tagging model
                POSModel model = new POSModel(modelIn);

                tokenizer = SimpleTokenizer.INSTANCE;
                tagger = new POSTaggerME(model);

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] tokens = tokenizer.tokenize(line);
            String[] tags = tagger.tag(tokens);

            for (int i = 0; i < tokens.length; i++) {
                context.write(new Text(tags[i]), new IntWritable(1));
            }
        }
    }

    public static class POSReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;

            for (IntWritable value : values) {
                sum += value.get();
            }

            context.write(key, new IntWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: PosTaggingPairs <input> <output>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "POS tagging using Pairs");

        job.setJarByClass(PosTaggingPairs.class);
        job.setMapperClass(POSMapper.class);
        job.setReducerClass(POSReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}


