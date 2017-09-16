import tensorflow as tf
from tensorflow.contrib import rnn
import csv
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-dir',
        required=True
    )
    parser.add_argument(
        '--train-file',
        required=True
    )

    parser.add_argument(
        '--job-name',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__


    pathToJobDir = arguments.pop('job_dir')
    jobName = arguments.pop('job_name')
    pathToData = arguments.pop('train_file')
    pathToLogDir = arguments.pop('log_dir')

    csv_file = pathToData
    log_dir = pathToLogDir

    hidden_size = 5
    batch_size = 1
    input_sequence_length = 10  
    output_sequence_length = 1
    input_num_classes = 10
    output_num_classes = 2
    stack = 2
    softmax_count = 1
    softmax_hidden_size = 5
    learning_rate = 0.1
    

    with tf.name_scope("placeholder") as scope:

        X = tf.placeholder(tf.float32, [None, input_sequence_length], name="x_input")
        #X_one_hot = tf.one_hot(X, input_num_classes)
        #print("X_one_hot", X_one_hot)  # check out the shape

        Y = tf.placeholder(tf.float32, [None, output_sequence_length], name="y_input")  # 1
        Y_one_hot = tf.one_hot(Y, output_num_classes)  # one hot
        print("Y_one_hot", Y_one_hot)
        Y_one_hot = tf.reshape(Y_one_hot, [-1, output_num_classes])
        print("Y_reshape", Y_one_hot)

        outputs = tf.reshape(X, [batch_size, input_sequence_length])


    with tf.name_scope("softmax_Layer_1") as scope:

        W1 = tf.Variable(tf.random_normal([input_sequence_length, softmax_hidden_size]), name='weight1')
        b1 = tf.Variable(tf.random_normal([softmax_hidden_size]), name='bias1')

        w1_hist = tf.summary.histogram("weights1", W1)
        b1_hist = tf.summary.histogram("biases1", b1)

        outputs = tf.matmul(outputs, W1) + b1


    with tf.name_scope("softmax_Layer_2") as scope:

        W2 = tf.Variable(tf.random_normal([softmax_hidden_size, output_num_classes]), name='weight2')
        b2 = tf.Variable(tf.random_normal([output_num_classes]), name='bias2')

        w2_hist = tf.summary.histogram("weights2", W2)
        b2_hist = tf.summary.histogram("biases2", b2)

        # tf.nn.softmax computes softmax activations
        logits = tf.matmul(outputs, W2) + b2
    

    with tf.name_scope("hypothesis") as scope:
        
        hypothesis = tf.nn.softmax(logits)


    with tf.name_scope("cost") as scope:
        # Cross entropy cost/loss
        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
        cost = tf.reduce_mean(cost_i)

        cost_summ = tf.summary.scalar('cost', cost)


    with tf.name_scope("accuracy") as scope:

        prediction = tf.argmax(hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summ = tf.summary.scalar('accuracy', accuracy)


    with tf.name_scope("train") as scope:
    
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


    with tf.name_scope("get_input_data") as scope:

        filename_queue = tf.train.string_input_producer([csv_file])
        key, value = tf.TextLineReader().read(filename_queue)

        input_list = []
        for l in range(input_sequence_length + output_sequence_length):
            input_list.append([1])

        data = tf.decode_csv(value, record_defaults=input_list)


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step=0
        while step < 700000:
            step+=1
            datas = sess.run(data)
        
            dataX = []
            dataY = []
            # if testing max value = dataX.append(datas[0:-2]) dataY.append(datas[-2:-1])
            # else if testing min value = dataX.append(datas[0:-2]) dataY.append(datas[-1:])
            dataX.append(datas[0:-1])
            dataY.append(datas[-1:])
            
            _, loss, accur, hypo, summary = sess.run([optimizer, cost, accuracy, hypothesis, merged], feed_dict={X: dataX, Y: dataY})

            writer.add_summary(summary, step)

            hypo_list=[]
            for j in range(len(hypo[0])):
                hypo_list.append(round(hypo[0][j], 2))

            print(step, dataX, dataY)
            print(loss, accur, hypo_list)


        coord.request_stop()
        coord.join(threads)

        saver = tf.train.Saver()
        model_file = os.path.join(pathToJobDir, jobName)
        saver.save(sess, model_file, global_step=0)


