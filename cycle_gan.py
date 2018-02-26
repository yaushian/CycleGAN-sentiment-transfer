from lib.seq2seq import seq2seq 
from lib.discriminator import discriminator
from lib.discriminator_X import discriminator_X
from lib.ops import *
from data_utils import *
from functools import reduce
import tensorflow as tf
import numpy as np
import os

class cycle_gan():

    def __init__(self,args,sess):
        
        self.sess = sess
        #mode can be training the whole model, autoencoder, gan, discriminator
        self.mode = args.mode
        
        #model config
        self.pretrain_discriminator_steps = 0
        self.discriminator_iterations = args.dis_iter
        self.do_gradient_penalty = True
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.num_steps =args.num_steps
        self.load_model = args.load
        self.saving_step = args.saving_step


        #trivial things
        self.lstm_length = [self.sequence_length+1 for _ in range(self.batch_size)] 
        self.utils = data_utils(args)

        #the model of generator, reconstructor, discriminator will be save in seperately directory
        self.model_dir = args.model_dir

        self.vocab_size = len(self.utils.id_word_dict)
        self.word_embedding_dim = 200
        self.BOS = self.utils.BOS_id
        self.EOS = self.utils.EOS_id

        self.build_model()


    def build_model(self):

        def get_all_variables():
            self.generator_X2Y_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator_X2Y") or v.name.startswith("generator_word")]
            for v in self.generator_X2Y_variables:
                print(v.name)
            self.generator_Y2X_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator_Y2X") or v.name.startswith("generator_word")]
            for v in self.generator_Y2X_variables:
                print(v.name)
            self.generator_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator")]

            self.discriminator_X_variables = [v for v in tf.trainable_variables() if v.name.startswith("discriminator_X") or v.name.startswith("discriminator_word")]
            for v in self.discriminator_X_variables:
                print(v.name)
            self.discriminator_Y_variables = [v for v in tf.trainable_variables() if v.name.startswith("discriminator_Y") or v.name.startswith("discriminator_word")]
            for v in self.discriminator_Y_variables:
                print(v.name)
            self.discriminator_variables = [v for v in tf.trainable_variables() if v.name.startswith("discriminator")]

            #get all saver with specific variable name
            self.generator_saver = tf.train.Saver(self.generator_variables,max_to_keep=6)
            self.discriminator_saver = tf.train.Saver(self.discriminator_variables,max_to_keep=2)


        def get_discriminator_loss(real_sample_score,false_sample_score,gradient_penalty):
            real_sample_score = tf.reduce_mean(real_sample_score)
            false_sample_score = tf.reduce_mean(false_sample_score)
            discriminator_loss = -(real_sample_score - false_sample_score) + 10.0*gradient_penalty
            
            return discriminator_loss


        def get_L2_loss(targets,outputs):
            assert targets.get_shape()==outputs.get_shape()
            
            shape = targets.get_shape().as_list()
            loss = tf.reduce_sum(tf.pow(targets - outputs, 2)) / (reduce(lambda x, y: x*y, shape))
            
            return loss


        def get_gradient_penalty(generator_outputs_embedded,real_sample_embedded,dis_fn):

            #set gradient penalty
            alpha = tf.random_uniform(
                shape=[self.batch_size,1,1], 
                minval=0.,
                maxval=1.)
            
            differences = generator_outputs_embedded - real_sample_embedded
            interpolates = real_sample_embedded + (alpha*differences)

            gradients = tf.gradients(dis_fn(interpolates),[interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)

            return gradient_penalty
            

        """
        the following code builds X to Y to X graph
        """
        def build_XYX_graph():
            with tf.variable_scope("XYX_inputs"):
                self.X2Y_inputs = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length))
                X2Y_inputs = tf.concat([self.X2Y_inputs,EOS_slice],axis=1)
                X2Y_inputs = tf.nn.embedding_lookup(word_embedding_matrix, X2Y_inputs)

                self.real_Y_sample = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,self.sequence_length))
                real_Y_sample = tf.concat([self.real_Y_sample,EOS_slice],axis=1)
                real_Y_sample = tf.nn.embedding_lookup(word_embedding_matrix, real_Y_sample)
                
                Y2X_decoder_inputs = tf.concat([BOS_slice,self.X2Y_inputs],axis=1)
                Y2X_decoder_inputs = tf.nn.embedding_lookup(word_embedding_matrix, Y2X_decoder_inputs)

                X2Y_decoder_inputs = tf.zeros([self.batch_size,self.sequence_length+1],dtype=tf.int32) + self.BOS
                if self.mode=='pretrain':
                    X2Y_decoder_inputs = tf.concat([BOS_slice,self.X2Y_inputs],axis=1)
                X2Y_decoder_inputs = tf.nn.embedding_lookup(word_embedding_matrix, X2Y_decoder_inputs)
                

            with tf.variable_scope("generator_X2Y") as scope:
                X2Y_outputs = seq2seq(
                    encoder_inputs = X2Y_inputs,
                    encoder_length = self.lstm_length,
                    decoder_inputs = X2Y_decoder_inputs,
                    word_embedding_dim = self.word_embedding_dim,
                    mode = self.mode
                )
                self.X2Y_test_outputs = X2Y_outputs


            with tf.variable_scope("generator_Y2X") as scope:
                Y2X_rec_outputs = seq2seq(
                    encoder_inputs = X2Y_outputs,
                    encoder_length = self.lstm_length,
                    decoder_inputs = Y2X_decoder_inputs,
                    word_embedding_dim = self.word_embedding_dim,
                    mode = self.mode
                )


            with tf.variable_scope("discriminator_Y") as scope:                
                false_Y_sample_score = discriminator(X2Y_outputs)
                self.false_Y_sample_score = tf.reduce_mean(false_Y_sample_score)

                scope.reuse_variables()

                real_Y_sample_score = discriminator(real_Y_sample)
                dis_Y_penalty = get_gradient_penalty(X2Y_outputs,real_Y_sample,discriminator)
                self.gpy = dis_Y_penalty


            with tf.variable_scope("XYX_loss") as scope:

                self.X2Y_reconstruction_loss = get_L2_loss(X2Y_inputs,Y2X_rec_outputs)

                self.discriminator_Y_loss = get_discriminator_loss(real_Y_sample_score,false_Y_sample_score,dis_Y_penalty)

                self.pretrain_X2Y_loss = get_L2_loss(X2Y_inputs,X2Y_outputs)

        """
        the following code builds Y to X to Y graph
        """
        def build_YXY_graph():
            with tf.variable_scope("YXY_inputs") as scope:
                self.Y2X_inputs = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.sequence_length))
                Y2X_inputs = tf.concat([self.Y2X_inputs,EOS_slice],axis=1)
                Y2X_inputs = tf.nn.embedding_lookup(word_embedding_matrix, Y2X_inputs)

                self.real_X_sample = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,self.sequence_length))
                real_X_sample = tf.concat([self.real_X_sample,EOS_slice],axis=1)
                real_X_sample = tf.nn.embedding_lookup(word_embedding_matrix, real_X_sample)
                
                X2Y_decoder_inputs = tf.concat([BOS_slice,self.Y2X_inputs],axis=1)
                X2Y_decoder_inputs = tf.nn.embedding_lookup(word_embedding_matrix, X2Y_decoder_inputs)

                Y2X_decoder_inputs = tf.zeros([self.batch_size,self.sequence_length+1],dtype=tf.int32) + self.BOS
                if self.mode=='pretrain':
                    Y2X_decoder_inputs = tf.concat([BOS_slice,self.Y2X_inputs],axis=1)
                Y2X_decoder_inputs = tf.nn.embedding_lookup(word_embedding_matrix, Y2X_decoder_inputs)


            with tf.variable_scope("generator_Y2X") as scope:
                scope.reuse_variables()
                Y2X_outputs = seq2seq(
                    encoder_inputs = Y2X_inputs,
                    encoder_length = self.lstm_length,
                    decoder_inputs = Y2X_decoder_inputs,
                    word_embedding_dim = self.word_embedding_dim,
                    mode = self.mode
                )
                self.Y2X_test_outputs = Y2X_outputs


            with tf.variable_scope("generator_X2Y") as scope:
                scope.reuse_variables()
                X2Y_rec_outputs = seq2seq(
                    encoder_inputs = Y2X_outputs,
                    encoder_length = self.lstm_length,
                    decoder_inputs = X2Y_decoder_inputs,
                    word_embedding_dim = self.word_embedding_dim,
                    mode = self.mode
                )

            with tf.variable_scope("discriminator_X") as scope:
                false_X_sample_score = discriminator_X(Y2X_outputs)
                self.false_X_sample_score = tf.reduce_mean(false_X_sample_score)

                scope.reuse_variables()

                real_X_sample_score = discriminator_X(real_X_sample)
                dis_X_penalty = get_gradient_penalty(Y2X_outputs,real_X_sample,discriminator_X)


            with tf.variable_scope("YXY_loss") as scope:

                self.Y2X_reconstruction_loss = get_L2_loss(Y2X_inputs,X2Y_rec_outputs)
                
                self.discriminator_X_loss = get_discriminator_loss(real_X_sample_score,false_X_sample_score,dis_X_penalty)

                self.pretrain_Y2X_loss = get_L2_loss(Y2X_inputs,Y2X_outputs)


        """
        the graph begin in here
        """

        BOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.BOS
        EOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.EOS

        #the word embedding are shared for generator
        with tf.variable_scope('word_embedding_matrix') as scope:
            init = tf.constant(self.utils.word_array)
            word_embedding_matrix = tf.get_variable(
                name="word_embedding_matrix",
                initializer=init,
                trainable = False
            )

        build_XYX_graph()
        build_YXY_graph()
        get_all_variables()

        with tf.variable_scope('generator_loss') as scope:
            self.generator_X2Y_loss = (self.Y2X_reconstruction_loss + self.X2Y_reconstruction_loss)*2.0 - self.false_Y_sample_score
            self.generator_Y2X_loss = (self.Y2X_reconstruction_loss + self.X2Y_reconstruction_loss)*2.0 - self.false_X_sample_score


        with tf.variable_scope('optimizer') as scope:
            self.train_discriminator_X_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.discriminator_X_loss, 
                var_list = self.discriminator_X_variables
            )

            self.train_discriminator_Y_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.discriminator_Y_loss, 
                var_list = self.discriminator_Y_variables
            )

            self.train_Y2X_op = tf.train.RMSPropOptimizer(0.0001).minimize(
                self.generator_Y2X_loss,
                var_list=self.generator_Y2X_variables
            )

            self.train_X2Y_op = tf.train.RMSPropOptimizer(0.0001).minimize(
                self.generator_X2Y_loss,
                var_list=self.generator_X2Y_variables
            )

            self.pretrain_X2Y_op = tf.train.AdamOptimizer().minimize(
                self.pretrain_X2Y_loss,
                var_list=self.generator_X2Y_variables
            )

            self.pretrain_Y2X_op = tf.train.AdamOptimizer().minimize(
                self.pretrain_Y2X_loss,
                var_list=self.generator_Y2X_variables
            )


    def pretrain(self):
        #use auto-encoder to pretrain the generator     
        step = 0
        saving_step = 1000
        summary_step = int(50)
        
        print('Start pretrain generator!!!!')

        model_dir = os.path.join(self.model_dir,'generator/')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir,'model')
        saver = self.generator_saver
        cur_loss = 0.0

        self.sess.run(tf.global_variables_initializer())

        for X_batch,Y_batch in self.utils.pretrain_generator_data_generator():
            step += 1

            #use two different optimizers to pretrain generator 
            feed_dict = {
                self.X2Y_inputs:X_batch,
                self.Y2X_inputs:Y_batch
            }
            _,_,loss_X2Y,loss_Y2X,tt = self.sess.run([self.pretrain_X2Y_op,self.pretrain_Y2X_op,self.pretrain_X2Y_loss,self.pretrain_Y2X_loss,self.Y2X_test_outputs],feed_dict=feed_dict)
            cur_loss += ( loss_X2Y + loss_Y2X ) / 2.0

            if step%(summary_step)==0:
                print(self.utils.vec2sent(tt[0]))
                print('{step}: generator_loss: {loss}'.format(step=step,loss=cur_loss/summary_step))
                cur_loss = 0.0


            if step%saving_step==0:
                print('saving.........')
                saver.save(self.sess, model_path, global_step=step)
                saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

            if step>=self.num_steps:
                break

        
    def train(self): 
        #train whole model

        print('Start training whole model!!')
        step = 0
        saving_step = 1000
        summary_step = int(50)
        #init model config
        self.sess.run(tf.global_variables_initializer())
            
        #discriminator model
        dis_model_dir = os.path.join(self.model_dir,'discriminator/')
        dis_model_path = os.path.join(dis_model_dir,'whole_model')
        if not os.path.exists(dis_model_dir):
            os.makedirs(dis_model_dir)

        #generator model
        gen_model_dir = os.path.join(self.model_dir,'generator/')
        gen_model_path = os.path.join(gen_model_dir,'whole_model')
        if not os.path.exists(gen_model_dir):
            os.makedirs(gen_model_dir)

        #init loss
        dis_X_loss = 0.0;dis_Y_loss = 0.0;xy_l = 0.0;yx_l = 0.0;xy_r = 0.0;yx_r = 0.0

        self.generator_saver.restore(self.sess,tf.train.latest_checkpoint(gen_model_dir))
        if self.load_model:
            print('load model from:',self.model_dir)
            self.discriminator_saver.restore(self.sess,tf.train.latest_checkpoint(dis_model_dir))

        for real_X_batches,real_Y_batches in self.utils.gan_data_generator():
            step += 1

            #train discriminator
            for i in range(self.discriminator_iterations):
                feed_dict = {
                    self.X2Y_inputs:real_X_batches[i+1],
                    self.Y2X_inputs:real_Y_batches[i+1],
                    self.real_X_sample:real_X_batches[i],
                    self.real_Y_sample:real_Y_batches[i]
                }
                #print(self.utils.id2sent(real_X_batches[i][0]))

                _,_,loss_X,loss_Y = self.sess.run(
                    [self.train_discriminator_X_op,self.train_discriminator_Y_op,\
                    self.discriminator_X_loss,self.discriminator_Y_loss],\
                    feed_dict=feed_dict)

                dis_X_loss += loss_X/self.discriminator_iterations
                dis_Y_loss += loss_Y/self.discriminator_iterations

            #train generator
            if step>=self.pretrain_discriminator_steps:
                #train generator only
                feed_dict = {
                    self.X2Y_inputs:real_X_batches[self.discriminator_iterations + 3],
                    self.Y2X_inputs:real_Y_batches[self.discriminator_iterations + 3]
                }
                
                _,_,l0,l1,r0,r1,pred = self.sess.run(
                    [self.train_X2Y_op,self.train_Y2X_op,\
                    self.generator_X2Y_loss,self.generator_Y2X_loss,\
                    self.X2Y_reconstruction_loss,self.Y2X_reconstruction_loss,self.X2Y_test_outputs],\
                    feed_dict=feed_dict)
                
                xy_l += l0
                yx_l += l1
                xy_r += r0
                yx_r += r1

                origin = real_X_batches[self.discriminator_iterations + 3]

            #make summary
            if step%(summary_step)==1:
                print('step: {step} dis_X_loss: {x_l} dis_Y_loss: {y_l} generator_X2Y_loss: {xy_l} generator_Y2X_loss {yx_l}'
                      ' X2Y_reconstruction_loss {xy_r} Y2X_reconstruction_loss {yx_r}'.format(\
                      x_l=dis_X_loss/summary_step,y_l=dis_Y_loss/summary_step,step=step,xy_l=xy_l/summary_step,\
                      yx_l=yx_l/summary_step,xy_r=xy_r/summary_step,yx_r=yx_r/summary_step))
                if step>self.pretrain_discriminator_steps:
                    print('origin_X:',self.utils.id2sent(origin[0]))
                    print('pred:',self.utils.vec2sent(pred[0]))
                print('')
                dis_X_loss = 0.0;dis_Y_loss = 0.0;xy_l = 0.0;yx_l = 0.0;xy_r = 0.0;yx_r = 0.0
                
            if step%saving_step==0:
                print('saving model!!!!......')
                self.discriminator_saver.save(self.sess, dis_model_path, global_step=step)
                self.generator_saver.save(self.sess, gen_model_path, global_step=step)

            if step>=self.num_steps:
                break


    def test(self):
        sentence = 'hi'
        gen_model_dir = os.path.join(self.model_dir,'generator/')
        self.sess.run(tf.global_variables_initializer())
        self.generator_saver.restore(self.sess, tf.train.latest_checkpoint(gen_model_dir))
        print('please enter one negative sentence')

        while(sentence):
            sentence = input('>')
            #sentence = sentence.split(':')[1]
            input_sent_vec = self.utils.sent2id(sentence)
            print(input_sent_vec)
            sent_vec = np.zeros((self.batch_size,self.sequence_length),dtype=np.int32)
            sent_vec[0] = input_sent_vec

            feed_dict = {
                    self.X2Y_inputs:sent_vec
            }
            preds = self.sess.run([self.X2Y_test_outputs],feed_dict)
            pred_sent = self.utils.vec2sent(preds[0][0])
            print(pred_sent)

    def file_test(self):
        line_count = 0
        out_fp = open('test_out.txt','w')
        gen_model_dir = os.path.join(self.model_dir,'generator/')
        self.sess.run(tf.global_variables_initializer())
        self.generator_saver.restore(self.sess, tf.train.latest_checkpoint(gen_model_dir))
        for test_batch in self.utils.test_data_generator():
            feed_dict = {self.X2Y_inputs:test_batch}
            preds = self.sess.run([self.X2Y_test_outputs],feed_dict)
            preds = preds[0]
            for pred in preds:
                out_fp.write(self.utils.vec2sent(pred) + '\n')
                line_count += 1
                if line_count>=28658:
                    break
            if line_count>=28658:
                break