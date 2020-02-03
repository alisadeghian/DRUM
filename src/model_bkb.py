import numpy as np 
import tensorflow as tf


class Learner(object):
    """
    This class builds a computation graph that represents the 
    neural ILP model and handles related graph running acitivies, 
    including update, predict, and get_attentions for given queries. 

    Args:
        option: hyper-parameters
    """

    def __init__(self, option):
        self.seed = option.seed        
        self.num_step = option.num_step
        self.rank = option.rank
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size
        
        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k
        
        self.num_entity = option.num_entity
        self.num_operator = option.num_operator
        self.query_is_language = option.query_is_language
        
        if not option.query_is_language:
            self.num_query = option.num_query
            self.query_embed_size = option.query_embed_size       
        else:
            self.vocab_embed_size = option.vocab_embed_size
            self.query_embed_size = self.vocab_embed_size
            self.num_vocab = option.num_vocab
            self.num_word = option.num_word    
        
        np.random.seed(self.seed)
        self._build_graph()

    def _random_uniform_unit(self, r, c):
        """ Initialize random and unit row norm matrix of size (r, c). """
        bound = 6./ np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(map(lambda row: row / np.linalg.norm(row), init_matrix))
        return init_matrix

    def _clip_if_not_None(self, g, v, low, high):
        """ Clip not-None gradients to (low, high). """
        """ Gradient of T is None if T not connected to the objective. """
        if g is not None:
            return (tf.clip_by_value(g, low, high), v)
        else:
            return (g, v)
    
    def _build_input(self):
        self.tails = tf.placeholder(tf.int32, [None])
        self.heads = tf.placeholder(tf.int32, [None])
        self.targets = tf.one_hot(indices=self.heads, depth=self.num_entity)
            
        if not self.query_is_language:
            self.queries = tf.placeholder(tf.int32, [None, self.num_step])
            self.query_embedding_params = tf.Variable(self._random_uniform_unit(
                                                          self.num_query + 1, # <END> token 
                                                          self.query_embed_size), 
                                                      dtype=tf.float32)
        
            rnn_inputs = tf.nn.embedding_lookup(self.query_embedding_params, 
                                                self.queries)
        else:
            self.queries = tf.placeholder(tf.int32, [None, self.num_step, self.num_word])
            self.vocab_embedding_params = tf.Variable(self._random_uniform_unit(
                                                          self.num_vocab + 1, # <END> token
                                                          self.vocab_embed_size),
                                                      dtype=tf.float32)
            embedded_query = tf.nn.embedding_lookup(self.vocab_embedding_params, 
                                                    self.queries)
            rnn_inputs = tf.reduce_mean(embedded_query, axis=2)

        return rnn_inputs


    def _build_graph(self):
        """ Build a computation graph that represents the model """
        rnn_inputs = self._build_input()                        
        # rnn_inputs: a list of num_step tensors,
        # each tensor of size (batch_size, query_embed_size).
        # self.rank=3
        if self.rank==1:    
            self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) 
                               for q in tf.split(rnn_inputs, 
                                                 self.num_step, 
                                                 axis=1)]
            self.rnn_inputs_new=[a for i, a in enumerate(self.rnn_inputs) if i< self.num_step-1]
            #print(len(self.rnn_inputs_new))
            
            cell = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell = tf.contrib.rnn.MultiRNNCell(
                                                        [cell] * self.num_layer, 
                                                        state_is_tuple=True)
            init_state = self.cell.zero_state(tf.shape(self.tails)[0], tf.float32)

            


            ##### making backward cells

            cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell_bw] * self.num_layer, 
                                                        state_is_tuple=True)
            init_state_bw = self.cell_bw.zero_state(tf.shape(self.tails)[0], tf.float32)




            

            
            # rnn_outputs: a list of num_step tensors,
            # each tensor of size (batch_size, rnn_state_size).
            self.rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                self.cell,
                self.cell_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state,
                initial_state_bw=init_state_bw,
                scope='f1')

            
            self.W_0 = tf.Variable(np.random.randn(
                                    self.rnn_state_size*2, 
                                    self.num_operator + 1),
                                dtype=tf.float32)
            self.b_0 = tf.Variable(np.zeros(
                                    (1, self.num_operator + 1)),
                                dtype=tf.float32)




            


            

            

            # attention_operators: a list of num_step lists,
            # each inner list has num_operator tensors,
            # each tensor of size (batch_size, 1).
            # Each tensor represents the attention over an operator.


            self.attention_operators = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output in self.rnn_outputs]

            

            # attention_memories: (will be) a list of num_step tensors,
            # each of size (batch_size, t+1),
            # where t is the current step (zero indexed).
            # Each tensor represents the attention over currently populated memory cells. 
            # self.attention_memories = []
            
            # memories: (will be) a tensor of size (batch_size, t+1, num_entity),
            # where t is the current step (zero indexed)
            # Then tensor represents currently populated memory cells.
            self.memories = tf.expand_dims(
                             tf.one_hot(
                                    indices=self.tails,
                                    depth=self.num_entity), 1)

            
            self.database = {r: tf.sparse_placeholder(
                                dtype=tf.float32, 
                                name="database_%d" % r)
                                for r in xrange(self.num_operator/2)}
            
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read = self.memories[:, -1, :]
                
                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []    
                    memory_read = tf.transpose(memory_read)
                    for r in xrange(self.num_operator/2):
                        for op_matrix, op_attn in zip(
                                        [self.database[r], 
                                         tf.sparse_transpose(self.database[r])],
                                        [self.attention_operators[t][r], 
                                         self.attention_operators[t][r+self.num_operator/2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read) * self.attention_operators[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))                
                    
                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories = tf.concat(
                        [self.memories, 
                        tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions1 = memory_read

            

            # -------------- end -------------------

            self.predictions = self.predictions1



        if self.rank==3:    
            self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) 
                               for q in tf.split(rnn_inputs, 
                                                 self.num_step, 
                                                 axis=1)]
            self.rnn_inputs_new=[a for i, a in enumerate(self.rnn_inputs) if i< self.num_step-1]
            #print(len(self.rnn_inputs_new))
            
            cell = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell = tf.contrib.rnn.MultiRNNCell(
                                                        [cell] * self.num_layer, 
                                                        state_is_tuple=True)
            init_state = self.cell.zero_state(tf.shape(self.tails)[0], tf.float32)

            cell2 = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell2 = tf.contrib.rnn.MultiRNNCell(
                                                        [cell2] * self.num_layer,
                                                        state_is_tuple=True)
            init_state2 = self.cell2.zero_state(tf.shape(self.tails)[0], tf.float32)


            cell3 = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell3 = tf.contrib.rnn.MultiRNNCell(
                                                        [cell3] * self.num_layer,
                                                        state_is_tuple=True)
            init_state3 = self.cell3.zero_state(tf.shape(self.tails)[0], tf.float32)




            ##### making backward cells

            cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell_bw] * self.num_layer, 
                                                        state_is_tuple=True)
            init_state_bw = self.cell_bw.zero_state(tf.shape(self.tails)[0], tf.float32)




            cell2_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell2_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell2_bw] * self.num_layer,
                                                        state_is_tuple=True)
            init_state2_bw = self.cell2_bw.zero_state(tf.shape(self.tails)[0], tf.float32)




            cell3_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell3_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell3_bw] * self.num_layer,
                                                        state_is_tuple=True)
            init_state3_bw = self.cell3_bw.zero_state(tf.shape(self.tails)[0], tf.float32)










            
            # rnn_outputs: a list of num_step tensors,
            # each tensor of size (batch_size, rnn_state_size).
            self.rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                self.cell,
                self.cell_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state,
                initial_state_bw=init_state_bw,
                scope='f1')

            self.rnn_outputs2, _, _= tf.contrib.rnn.static_bidirectional_rnn(
                self.cell2,
                self.cell2_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state2,
                initial_state_bw=init_state2_bw,
                scope='f2')

            self.rnn_outputs3, _, _= tf.contrib.rnn.static_bidirectional_rnn(
                self.cell3,
                self.cell3_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state3,
                initial_state_bw=init_state3_bw,
                scope='f3')
            # making a NN here with 128 hidden units and leaky relu
            

            self.W_0 = tf.Variable(np.random.randn(
                                    self.rnn_state_size*2, 
                                    self.num_operator + 1),
                                dtype=tf.float32)
            self.b_0 = tf.Variable(np.zeros(
                                    (1, self.num_operator + 1)),
                                dtype=tf.float32)




            


            

            

            # attention_operators: a list of num_step lists,
            # each inner list has num_operator tensors,
            # each tensor of size (batch_size, 1).
            # Each tensor represents the attention over an operator.


            self.attention_operators = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output in self.rnn_outputs]

            self.attention_operators2 = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output2, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output2 in self.rnn_outputs2]

            self.attention_operators3 = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output3, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output3 in self.rnn_outputs3]

            # attention_memories: (will be) a list of num_step tensors,
            # each of size (batch_size, t+1),
            # where t is the current step (zero indexed).
            # Each tensor represents the attention over currently populated memory cells. 
            # self.attention_memories = []
            
            # memories: (will be) a tensor of size (batch_size, t+1, num_entity),
            # where t is the current step (zero indexed)
            # Then tensor represents currently populated memory cells.
            self.memories = tf.expand_dims(
                             tf.one_hot(
                                    indices=self.tails,
                                    depth=self.num_entity), 1)

            self.memories2 = tf.expand_dims(
                            tf.one_hot(
                                indices=self.tails,
                                depth=self.num_entity), 1)
            self.memories3 = tf.expand_dims(
                            tf.one_hot(
                                indices=self.tails,
                                depth=self.num_entity), 1)

            self.database = {r: tf.sparse_placeholder(
                                dtype=tf.float32, 
                                name="database_%d" % r)
                                for r in xrange(self.num_operator/2)}
            
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read = self.memories[:, -1, :]
                
                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []    
                    memory_read = tf.transpose(memory_read)
                    for r in xrange(self.num_operator/2):
                        for op_matrix, op_attn in zip(
                                        [self.database[r], 
                                         tf.sparse_transpose(self.database[r])],
                                        [self.attention_operators[t][r], 
                                         self.attention_operators[t][r+self.num_operator/2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read) * self.attention_operators[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))                
                    
                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories = tf.concat(
                        [self.memories, 
                        tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions1 = memory_read

            # -------------- second for ------------
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read2 = self.memories2[:, -1, :]

                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []
                    memory_read2 = tf.transpose(memory_read2)
                    for r in xrange(self.num_operator / 2):
                        for op_matrix, op_attn in zip(
                                [self.database[r],
                                 tf.sparse_transpose(self.database[r])],
                                [self.attention_operators2[t][r],
                                 self.attention_operators2[t][r + self.num_operator / 2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read2)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read2) * self.attention_operators2[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1,
                                                                                     keep_dims=True))

                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1. - self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories2 = tf.concat(
                        [self.memories2,
                         tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions2 = memory_read2

            # -------------- third for ------------
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read3 = self.memories3[:, -1, :]

                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []
                    memory_read3 = tf.transpose(memory_read3)
                    for r in xrange(self.num_operator / 2):
                        for op_matrix, op_attn in zip(
                                [self.database[r],
                                 tf.sparse_transpose(self.database[r])],
                                [self.attention_operators3[t][r],
                                 self.attention_operators3[t][r + self.num_operator / 2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read3)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read3) * self.attention_operators3[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1,
                                                                                     keep_dims=True))

                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1. - self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories3 = tf.concat(
                        [self.memories3,
                         tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions3 = memory_read3






            # -------------- end -------------------

            self.predictions = self.predictions1 + self.predictions2+self.predictions3
        if self.rank==4:
                
            self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) 
                               for q in tf.split(rnn_inputs, 
                                                 self.num_step, 
                                                 axis=1)]
            self.rnn_inputs_new=[a for i, a in enumerate(self.rnn_inputs) if i< self.num_step-1]
            #print(len(self.rnn_inputs_new))
            
            cell = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell = tf.contrib.rnn.MultiRNNCell(
                                                        [cell] * self.num_layer, 
                                                        state_is_tuple=True)
            init_state = self.cell.zero_state(tf.shape(self.tails)[0], tf.float32)

            cell2 = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell2 = tf.contrib.rnn.MultiRNNCell(
                                                        [cell2] * self.num_layer,
                                                        state_is_tuple=True)
            init_state2 = self.cell2.zero_state(tf.shape(self.tails)[0], tf.float32)


            cell3 = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell3 = tf.contrib.rnn.MultiRNNCell(
                                                        [cell3] * self.num_layer,
                                                        state_is_tuple=True)
            init_state3 = self.cell3.zero_state(tf.shape(self.tails)[0], tf.float32)



            cell4 = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell4 = tf.contrib.rnn.MultiRNNCell(
                                                        [cell4] * self.num_layer,
                                                        state_is_tuple=True)
            init_state4 = self.cell4.zero_state(tf.shape(self.tails)[0], tf.float32)




            ##### making backward cells

            cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell_bw] * self.num_layer, 
                                                        state_is_tuple=True)
            init_state_bw = self.cell_bw.zero_state(tf.shape(self.tails)[0], tf.float32)




            cell2_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell2_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell2_bw] * self.num_layer,
                                                        state_is_tuple=True)
            init_state2_bw = self.cell2_bw.zero_state(tf.shape(self.tails)[0], tf.float32)




            cell3_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell3_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell3_bw] * self.num_layer,
                                                        state_is_tuple=True)
            init_state3_bw = self.cell3_bw.zero_state(tf.shape(self.tails)[0], tf.float32)




            cell4_bw = tf.contrib.rnn.LSTMCell(self.rnn_state_size)
            self.cell4_bw = tf.contrib.rnn.MultiRNNCell(
                                                        [cell4_bw] * self.num_layer,
                                                        state_is_tuple=True)
            init_state4_bw = self.cell4_bw.zero_state(tf.shape(self.tails)[0], tf.float32)










            
            # rnn_outputs: a list of num_step tensors,
            # each tensor of size (batch_size, rnn_state_size).
            self.rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                self.cell,
                self.cell_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state,
                initial_state_bw=init_state_bw,
                scope='f1')

            self.rnn_outputs2, _, _= tf.contrib.rnn.static_bidirectional_rnn(
                self.cell2,
                self.cell2_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state2,
                initial_state_bw=init_state2_bw,
                scope='f2')

            self.rnn_outputs3, _, _= tf.contrib.rnn.static_bidirectional_rnn(
                self.cell3,
                self.cell3_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state3,
                initial_state_bw=init_state3_bw,
                scope='f3')



            self.rnn_outputs4, _, _= tf.contrib.rnn.static_bidirectional_rnn(
                self.cell4,
                self.cell4_bw,
                self.rnn_inputs_new,
                initial_state_fw=init_state4,
                initial_state_bw=init_state4_bw,
                scope='f4')
            # making a NN here with 128 hidden units and leaky relu
            

            self.W_0 = tf.Variable(np.random.randn(
                                    self.rnn_state_size*2, 
                                    self.num_operator + 1),
                                dtype=tf.float32)
            self.b_0 = tf.Variable(np.zeros(
                                    (1, self.num_operator + 1)),
                                dtype=tf.float32)




            


            

            

            # attention_operators: a list of num_step lists,
            # each inner list has num_operator tensors,
            # each tensor of size (batch_size, 1).
            # Each tensor represents the attention over an operator.


            self.attention_operators = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output in self.rnn_outputs]

            self.attention_operators2 = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output2, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output2 in self.rnn_outputs2]

            self.attention_operators3 = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output3, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output3 in self.rnn_outputs3]
            self.attention_operators4 = [tf.split(
                                        tf.nn.softmax(
                                        
                                            tf.matmul(rnn_output4, self.W_0) + self.b_0),
                                            
                                        self.num_operator + 1,
                                        axis=1) 
                                        for rnn_output4 in self.rnn_outputs4]

            # attention_memories: (will be) a list of num_step tensors,
            # each of size (batch_size, t+1),
            # where t is the current step (zero indexed).
            # Each tensor represents the attention over currently populated memory cells. 
            # self.attention_memories = []
            
            # memories: (will be) a tensor of size (batch_size, t+1, num_entity),
            # where t is the current step (zero indexed)
            # Then tensor represents currently populated memory cells.
            self.memories = tf.expand_dims(
                             tf.one_hot(
                                    indices=self.tails,
                                    depth=self.num_entity), 1)

            self.memories2 = tf.expand_dims(
                            tf.one_hot(
                                indices=self.tails,
                                depth=self.num_entity), 1)
            self.memories3 = tf.expand_dims(
                            tf.one_hot(
                                indices=self.tails,
                                depth=self.num_entity), 1)
            self.memories4 = tf.expand_dims(
                            tf.one_hot(
                                indices=self.tails,
                                depth=self.num_entity), 1)

            self.database = {r: tf.sparse_placeholder(
                                dtype=tf.float32, 
                                name="database_%d" % r)
                                for r in xrange(self.num_operator/2)}
            
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read = self.memories[:, -1, :]
                
                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []    
                    memory_read = tf.transpose(memory_read)
                    for r in xrange(self.num_operator/2):
                        for op_matrix, op_attn in zip(
                                        [self.database[r], 
                                         tf.sparse_transpose(self.database[r])],
                                        [self.attention_operators[t][r], 
                                         self.attention_operators[t][r+self.num_operator/2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read) * self.attention_operators[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))                
                    
                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories = tf.concat(
                        [self.memories, 
                        tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions1 = memory_read

            # -------------- second for ------------
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read2 = self.memories2[:, -1, :]

                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []
                    memory_read2 = tf.transpose(memory_read2)
                    for r in xrange(self.num_operator / 2):
                        for op_matrix, op_attn in zip(
                                [self.database[r],
                                 tf.sparse_transpose(self.database[r])],
                                [self.attention_operators2[t][r],
                                 self.attention_operators2[t][r + self.num_operator / 2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read2)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read2) * self.attention_operators2[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1,
                                                                                     keep_dims=True))

                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1. - self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories2 = tf.concat(
                        [self.memories2,
                         tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions2 = memory_read2

            # -------------- third for ------------
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read3 = self.memories3[:, -1, :]

                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []
                    memory_read3 = tf.transpose(memory_read3)
                    for r in xrange(self.num_operator / 2):
                        for op_matrix, op_attn in zip(
                                [self.database[r],
                                 tf.sparse_transpose(self.database[r])],
                                [self.attention_operators3[t][r],
                                 self.attention_operators3[t][r + self.num_operator / 2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read3)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read3) * self.attention_operators3[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1,
                                                                                     keep_dims=True))

                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1. - self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories3 = tf.concat(
                        [self.memories3,
                         tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions3 = memory_read3




            # -------------- forth for ------------
            for t in xrange(self.num_step):
                # memory_read: tensor of size (batch_size, num_entity)
                # memory_read = tf.squeeze(self.memories, squeeze_dims=[1])
                memory_read4 = self.memories4[:, -1, :]

                if t < self.num_step - 1:
                    # database_results: (will be) a list of num_operator tensors,
                    # each of size (batch_size, num_entity).
                    database_results = []
                    memory_read4 = tf.transpose(memory_read4)
                    for r in xrange(self.num_operator / 2):
                        for op_matrix, op_attn in zip(
                                [self.database[r],
                                 tf.sparse_transpose(self.database[r])],
                                [self.attention_operators4[t][r],
                                 self.attention_operators4[t][r + self.num_operator / 2]]):
                            product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read4)
                            database_results.append(tf.transpose(product) * op_attn)
                    database_results.append(tf.transpose(memory_read4) * self.attention_operators4[t][-1])

                    added_database_results = tf.add_n(database_results)
                    if self.norm:
                        added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1,
                                                                                     keep_dims=True))

                    if self.dropout > 0.:
                        added_database_results = tf.nn.dropout(added_database_results, keep_prob=1. - self.dropout)

                    # Populate a new cell in memory by concatenating.
                    self.memories4 = tf.concat(
                        [self.memories4,
                         tf.expand_dims(added_database_results, 1)],
                        axis=1)
                else:
                    self.predictions4 = memory_read4
                    






            # -------------- end -------------------

            self.predictions = self.predictions1 + self.predictions2+self.predictions3+self.predictions4
        print(self.rank)







        self.final_loss = - tf.reduce_sum(self.targets * tf.log(tf.maximum(self.predictions, self.thr)), 1)
        
        if not self.accuracy:
            self.in_top = tf.nn.in_top_k(
                            predictions=self.predictions, 
                            targets=self.heads, 
                            k=self.top_k)
        else: 
            _, indices = tf.nn.top_k(self.predictions, self.top_k, sorted=False)
            self.in_top = tf.equal(tf.squeeze(indices), self.heads)
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        capped_gvs = map(
            lambda (grad, var): self._clip_if_not_None(grad, var, -5., 5.), gvs) 
        self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)

    def _run_graph(self, sess, qq, hh, tt, mdb, to_fetch):
        feed = {}
        if not self.query_is_language:
            feed[self.queries] = [[q] * (self.num_step-1) + [self.num_query]
                                  for q in qq]
        else:
            feed[self.queries] = [[q] * (self.num_step-1) 
                                  + [[self.num_vocab] * self.num_word]
                                  for q in qq]

        feed[self.heads] = hh 
        feed[self.tails] = tt 
        for r in xrange(self.num_operator / 2):
            feed[self.database[r]] = tf.SparseTensorValue(*mdb[r])
        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output

    def update(self, sess, qq, hh, tt, mdb):
        to_fetch = [self.final_loss, self.in_top, self.optimizer_step]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch) 
        return fetched[0], fetched[1]

    def predict(self, sess, qq, hh, tt, mdb):
        to_fetch = [self.final_loss, self.in_top]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return fetched[0], fetched[1]

    def get_predictions_given_queries(self, sess, qq, hh, tt, mdb):
        to_fetch = [self.in_top, self.predictions]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return fetched[0], fetched[1]

    def get_attentions_given_queries(self, sess, queries):
        qq = queries
        hh = [0] * len(queries)
        tt = [0] * len(queries)
        mdb = {r: ([(0,0)], [0.], (self.num_entity, self.num_entity)) 
                for r in xrange(self.num_operator / 2)}
        to_fetch = [self.attention_operators, self.attention_memories]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return fetched[0], fetched[1]

    def get_vocab_embedding(self, sess):
        qq = [[0] * self.num_word]
        hh = [0] * len(qq)
        tt = [0] * len(hh)
        mdb = {r: ([(0,0)], [0.], (self.num_entity, self.num_entity)) 
                for r in xrange(self.num_operator / 2)}
        to_fetch = self.vocab_embedding_params
        vocab_embedding = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return vocab_embedding
