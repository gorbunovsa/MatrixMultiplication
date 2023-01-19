import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class mm {
    private static final String DELIMITER_TAB = "\t";
    private static final String DELIMITER_COMMA = ",";
    private static int mmGroups;
    private static String mmTags;
    private static String mmFloatFormat;
    private static int mmNumReducers;
    private static int matrixSize;
    private static int groupSize;

    public static class mmMapper
            extends Mapper<Object,Text,Text,Text> {

        public void read_config(Context context) {
            Configuration conf = context.getConfiguration();
            mmGroups = conf.getInt("mm.groups", 1);
            mmTags = conf.get("mm.tags", "ABC");
            mmFloatFormat = conf.get("mm.float-format", "%.3f");
            mmNumReducers = conf.getInt("mapred.reduce.tasks", 1);
            matrixSize = conf.getInt("matrix.size", 1);
            groupSize = conf.getInt("group.size", 1);
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            read_config(context);
            String[] tokens = value.toString().split(DELIMITER_TAB);
            String key_out;
            int row_group = Integer.parseInt(tokens[1]) / groupSize,
                col_group = Integer.parseInt(tokens[2]) / groupSize;
            String row_col = row_group + DELIMITER_COMMA + col_group;
            for (int g = 0; g < mmGroups; g++) {
                if (tokens[0].equals(Character.toString(mmTags.charAt(0)))) {
                    key_out = row_col + DELIMITER_COMMA + g;
                } else {
                    key_out = g + DELIMITER_COMMA + row_col;
                }
                context.write(
                        new Text(key_out),
                        new Text(tokens[0] + DELIMITER_COMMA + tokens[1] + DELIMITER_COMMA + tokens[2] + DELIMITER_COMMA + tokens[3])
                );
            }
        }
    }

    public static class IdentityMapper
            extends Mapper<Object,Text,Text,Text> {
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] value_tokens = value.toString().split(DELIMITER_TAB);
            context.write(new Text(value_tokens[0]), new Text(value_tokens[1]));
        }
    }

    public static class MultReducer
            extends Reducer<Text,Text,Text,Text> {

        public void read_config(Context context) {
            Configuration conf = context.getConfiguration();
            mmGroups = conf.getInt("mm.groups", 1);
            mmTags = conf.get("mm.tags", "ABC");
            mmFloatFormat = conf.get("mm.float-format", "%.3f");
            mmNumReducers = conf.getInt("mapred.reduce.tasks", 1);
            matrixSize = conf.getInt("matrix.size", 1);
            groupSize = conf.getInt("group.size", 1);
        }

        public int getMaxIndex(int group_num) {
            if (matrixSize % groupSize != 0 & group_num + 1 == mmGroups) {
                return matrixSize % groupSize;
            }
            return groupSize;
        }

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            read_config(context);
            String[] key_tokens = key.toString().split(DELIMITER_COMMA);
            int I_group_num = Integer.parseInt(key_tokens[0]),
                J_group_num = Integer.parseInt(key_tokens[1]),
                K_group_num = Integer.parseInt(key_tokens[2]),
                i_max = getMaxIndex(I_group_num),
                j_max = getMaxIndex(J_group_num),
                k_max = getMaxIndex(K_group_num),
                cur_i,
                cur_j,
                cur_k;
            double[][] matrixA = new double[i_max][j_max];
            double[][] matrixB = new double[j_max][k_max];
            for (Text value : values) {
                String[] value_tokens = value.toString().split(DELIMITER_COMMA);
                if (value_tokens[0].equals(Character.toString(mmTags.charAt(0)))) {
                    cur_i = Integer.parseInt(value_tokens[1]) - I_group_num * groupSize;
                    cur_j = Integer.parseInt(value_tokens[2]) - J_group_num * groupSize;
                    matrixA[cur_i][cur_j] = Double.parseDouble(value_tokens[3]);
                } else {
                    cur_j = Integer.parseInt(value_tokens[1]) - J_group_num * groupSize;
                    cur_k = Integer.parseInt(value_tokens[2]) - K_group_num * groupSize;
                    matrixB[cur_j][cur_k] = Double.parseDouble(value_tokens[3]);
                }
            }
            double res;
            int i_global, k_global;
            for(int i = 0; i < i_max; i++) {
                for(int k = 0; k < k_max; k++) {
                    res = 0.0;
                    for(int j = 0; j < j_max; j++) {
                        res += matrixA[i][j]*matrixB[j][k];
                    }
                    if (res != 0.0) {
                        i_global = i + I_group_num*groupSize;
                        k_global = k + K_group_num*groupSize;
                        context.write(
                                new Text(i_global + DELIMITER_COMMA + k_global),
                                new Text(Double.toString(res))
                        );
                    }
                }
            }
        }
    }

    public static class SumReducer
            extends Reducer<Text,Text,Text,Text> {

        public void read_config(Context context) {
            Configuration conf = context.getConfiguration();
            mmGroups = conf.getInt("mm.groups", 1);
            mmTags = conf.get("mm.tags", "ABC");
            mmFloatFormat = conf.get("mm.float-format", "%.3f");
            mmNumReducers = conf.getInt("mapred.reduce.tasks", 1);
            matrixSize = conf.getInt("matrix.size", 1);
            groupSize = conf.getInt("group.size", 1);
        }

        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            read_config(context);
            double sum = 0.0;
            for (Text value : values){
                sum += Double.parseDouble(value.toString());
            }
            String[] key_tokens = key.toString().split(DELIMITER_COMMA);
            if (sum != 0.0) {
                context.write(new Text(mmTags.charAt(2) + DELIMITER_TAB + key_tokens[0] + DELIMITER_TAB + key_tokens[1]),
                        new Text(String.format(mmFloatFormat, sum)));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 3) {
            System.err.println("Usage: mm <in> <out>");
            System.exit(2);
        }

        mmNumReducers = conf.getInt("mapred.reduce.tasks", 1);
        mmGroups = conf.getInt("mm.groups", 1);

        FileSystem fs = FileSystem.get(conf);

        Path pt_size_input = new Path(otherArgs[0] + "/size");
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt_size_input)));
        String input_sizes = br.readLine();
        matrixSize = Integer.parseInt(input_sizes.split(DELIMITER_TAB)[0]);
        groupSize = (mmGroups < matrixSize) ? (int) Math.ceil((float) matrixSize / mmGroups) : 1;
        conf.setInt("matrix.size", matrixSize);
        conf.setInt("group.size", groupSize);

        String temp_path = otherArgs[2] + "/temp";

        Job job1 = new Job(conf, "mm1");

        job1.setJarByClass(mm.class);

        job1.setMapperClass(mmMapper.class);
        job1.setReducerClass(MultReducer.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        job1.setNumReduceTasks(mmNumReducers);

        MultipleInputs.addInputPath(job1, new Path(otherArgs[0] + "/data"), TextInputFormat.class, mmMapper.class);
        MultipleInputs.addInputPath(job1, new Path(otherArgs[1] + "/data"), TextInputFormat.class, mmMapper.class);
        FileOutputFormat.setOutputPath(job1, new Path(temp_path));

        job1.waitForCompletion(true);

        Job job2 = new Job(conf, "mm2");

        job2.setJarByClass(mm.class);

        job2.setMapperClass(IdentityMapper.class);
        job2.setReducerClass(SumReducer.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        job2.setNumReduceTasks(mmNumReducers);

        FileInputFormat.addInputPath(job2, new Path(temp_path));
        FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2] + "/data"));

        job2.waitForCompletion(true);

        Path pt_size_output = new Path(otherArgs[2] + "/size");
        FSDataOutputStream fsDataOutputStream = fs.create(pt_size_output,true);
        BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(fsDataOutputStream));
        bufferedWriter.write(matrixSize + DELIMITER_TAB + matrixSize);
        bufferedWriter.close();

        fs.delete(new Path(temp_path), true);

        fs.close();
    }
}