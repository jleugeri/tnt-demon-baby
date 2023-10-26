import unittest
import numpy as np
from ttt_pyttt import PyTTT

class TestPyTTT(unittest.TestCase):
    def setUp(self) -> None:
        self.NUM_PROCESSORS = 10
        
        duration = np.zeros(self.NUM_PROCESSORS, dtype=int)
        good_threshold = np.zeros(self.NUM_PROCESSORS)
        bad_threshold = np.zeros(self.NUM_PROCESSORS)
        W_good = np.zeros((self.NUM_PROCESSORS,self.NUM_PROCESSORS), dtype=int)
        W_bad = np.zeros((self.NUM_PROCESSORS,self.NUM_PROCESSORS), dtype=int)
        
        self.proc = PyTTT(good_threshold, bad_threshold, W_good, W_bad, duration)
    
    def test_init_parameters(self):
        # write parameters
        for i in range(self.NUM_PROCESSORS):
            pp = self.proc[i].parameters
            pp.badTokenThreshold = i
            pp.goodTokenThreshold =i
            pp.duration = i

        # check that all parameters were set correctly
        self.assertSequenceEqual(self.proc.goodTokenThreshold.tolist(), list(range(self.NUM_PROCESSORS)))
        self.assertSequenceEqual(self.proc.badTokenThreshold.tolist(), list(range(self.NUM_PROCESSORS)))
        self.assertSequenceEqual(self.proc.duration.tolist(), list(range(self.NUM_PROCESSORS)))

        # check that the parameters can be read back correctly
        for i in range(self.NUM_PROCESSORS):
            self.assertEqual(self.proc.goodTokenThreshold[i], i)
            self.assertEqual(self.proc.badTokenThreshold[i], i)
            self.assertEqual(self.proc.duration[i], i)
    
    def test_init_connections(self):
        W_good = np.random.poisson(1,(self.NUM_PROCESSORS,self.NUM_PROCESSORS))
        W_bad = np.random.poisson(1,(self.NUM_PROCESSORS,self.NUM_PROCESSORS))

        # check that setting the incoming connections works
        for i in range(self.NUM_PROCESSORS):
            proc_i = self.proc[i]
            # set all at once
            proc_i.connectionsIn.W_good = W_good[i,:]

            # set element-by-element
            for j in range(self.NUM_PROCESSORS):
                proc_i.connectionsIn.W_bad[j] = W_bad[i,j]
        
        # check that the connections were written properly
        self.assertSequenceEqual(self.proc.W_good.tolist(), W_good.tolist())
        self.assertSequenceEqual(self.proc.W_bad.tolist(), W_bad.tolist())

        # check that the connections can be read back correctly
        for i in range(self.NUM_PROCESSORS):
            proc_i = self.proc[i]
            self.assertSequenceEqual(proc_i.connectionsIn.W_good.tolist(), W_good[i,:].tolist())
            self.assertSequenceEqual(proc_i.connectionsIn.W_bad.tolist(), W_bad[i,:].tolist())

        # clear weights again and try to write via outgoing connections instead
        self.proc.W_good[:] = 0
        self.proc.W_bad[:] = 0

        # check that setting the outgoing connections works
        for i in range(self.NUM_PROCESSORS):
            proc_i = self.proc[i]
            # set all at once
            proc_i.connectionsOut.W_bad = W_bad[:,i]

            # set element-by-element
            for j in range(self.NUM_PROCESSORS):
                proc_i.connectionsOut.W_good[j] = W_good[j,i]
        
        # check that the connections were written properly
        self.assertSequenceEqual(self.proc.W_good.tolist(), W_good.tolist())
        self.assertSequenceEqual(self.proc.W_bad.tolist(), W_bad.tolist())

        # check that the connections can be read back correctly
        for i in range(self.NUM_PROCESSORS):
            proc_i = self.proc[i]
            self.assertSequenceEqual(proc_i.connectionsOut.W_good.tolist(), W_good[:,i].tolist())
            self.assertSequenceEqual(proc_i.connectionsOut.W_bad.tolist(), W_bad[:,i].tolist())
    
    def test_set_expiration_circular(self):
        NUM_SAMPLES = 100
        inp = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        duration = np.arange(0, self.NUM_PROCESSORS)

        # start a token at time t=10
        inp[10,:] = 1
        inp[95,:] = 1

        # set the expiration time 0,1,...,9 steps
        PyTTT.set_expiration(inp, duration+NUM_SAMPLES, circular=True)

        # check that every token added is later removed at the correct time
        for i in range(self.NUM_PROCESSORS):
            correct = np.zeros(100, dtype=int)
            correct[10] = 1
            correct[95] = 1
            correct[(10+1+duration[i]) % NUM_SAMPLES] = -1
            correct[(95+1+duration[i]) % NUM_SAMPLES] = -1
            self.assertSequenceEqual(inp[:,i].tolist(), correct.tolist())
    
    def test_set_expiration_noncircular(self):
        NUM_SAMPLES = 100
        inp = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        duration = np.arange(0, self.NUM_PROCESSORS)

        # start a token at time t=10 and one at t=95
        inp[10,:] = 1
        inp[95,:] = 1

        # set the expiration time 1,2,...,10 steps
        PyTTT.set_expiration(inp, duration, circular=False)

        # check that every token added is later removed at the correct time
        for i in range(self.NUM_PROCESSORS):
            correct = np.zeros(NUM_SAMPLES, dtype=int)
            correct[10] = 1
            correct[95] = 1
            if 10+1+duration[i] < NUM_SAMPLES:
                correct[10+1+duration[i]] = -1
            if 95+1+duration[i] < NUM_SAMPLES:
                correct[95+1+duration[i]] = -1
            self.assertSequenceEqual(inp[:,i].tolist(), correct.tolist())

        # now try again with wrap around (should not change anything)

        # start a token at time t=10
        inp[:,:] = 0
        inp[10,:] = 1
        inp[95,:] = 1

        # set the expiration time 1,2,...,10 steps
        PyTTT.set_expiration(inp, duration+NUM_SAMPLES, circular=False)

        # check that every token added is later removed at the correct time
        for i in range(self.NUM_PROCESSORS):
            correct = np.zeros(NUM_SAMPLES, dtype=int)
            correct[10] = 1
            correct[95] = 1
            self.assertSequenceEqual(inp[:,i].tolist(), correct.tolist())


    def test_processor_input_tracking(self):
        NUM_SAMPLES = 100
        DELAY = np.arange(1, self.NUM_PROCESSORS+1)
        
        # set inputs, record states and outputs
        # generate a random number of incoming tokens for each processor
        my_good_tokens_in = np.random.poisson(1, size=(NUM_SAMPLES, self.NUM_PROCESSORS))
        my_bad_tokens_in = np.random.poisson(1, size=(NUM_SAMPLES, self.NUM_PROCESSORS))

        # give each token a lifetime of DELAY
        PyTTT.set_expiration(my_good_tokens_in, DELAY)
        PyTTT.set_expiration(my_bad_tokens_in, DELAY)

        # precompute how many tokens there should be at each time step
        my_good_tokens_in_cum = np.cumsum(my_good_tokens_in, axis=0)
        my_bad_tokens_in_cum = np.cumsum(my_bad_tokens_in, axis=0)

        # run processors, iterating over the inputs
        for step,(start_token, stop_token) in enumerate(self.proc.run(my_good_tokens_in, my_bad_tokens_in)):
            # check that the inputs are tracked correctly
            self.assertSequenceEqual(self.proc.goodTokensIn.tolist(), my_good_tokens_in[step,:].tolist())
            self.assertSequenceEqual(self.proc.badTokensIn.tolist(), my_bad_tokens_in[step,:].tolist())

            # check that the token counters are correct
            self.assertSequenceEqual(self.proc.goodTokens.tolist(), my_good_tokens_in_cum[step,:].tolist())
            self.assertSequenceEqual(self.proc.badTokens.tolist(), my_bad_tokens_in_cum[step,:].tolist())

    def test_processor_pulse_generation(self):
        NUM_SAMPLES = 100
        INP_PULSE_LENGTH = 10

        duration = np.arange(0, self.NUM_PROCESSORS)
        threshold = np.arange(1, self.NUM_PROCESSORS+1)

        # set parameters incrementally
        for i in range(self.NUM_PROCESSORS):
            pp = self.proc[i].parameters
            pp.goodTokenThreshold = threshold[i]
            pp.badTokenThreshold = 1
            pp.duration = duration[i]

        # set inputs, record states and outputs
        # generate constant good token inputs
        my_good_tokens_in = np.ones((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        my_bad_tokens_in = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)

        # give each token a lifetime of DELAY
        PyTTT.set_expiration(my_good_tokens_in, INP_PULSE_LENGTH, circular=False)
        PyTTT.set_expiration(my_bad_tokens_in, INP_PULSE_LENGTH, circular=False)

        # each processor should generate a first pulse after its specific threshold has been reached,
        # and then fire periodically with a period equal to its specific duration
        should_start = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        should_stop = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        for i in range(self.NUM_PROCESSORS):
            start_times = np.arange(threshold[i]-1, NUM_SAMPLES, 2+duration[i])
            stop_times = start_times + duration[i] + 1
            stop_times = stop_times[stop_times < NUM_SAMPLES]
            should_start[start_times,i] = 1
            should_stop[stop_times,i] = 1


        # run processors, iterating over the inputs
        did_start = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        did_stop = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        for step,(start_token, stop_token) in enumerate(self.proc.run(my_good_tokens_in, my_bad_tokens_in)):
            did_start[step,:] = start_token
            did_stop[step,:] = stop_token

        # check that the pulse generation works correctly
        self.assertSequenceEqual(did_start.tolist(), should_start.tolist())
        self.assertSequenceEqual(did_stop.tolist(), should_stop.tolist())

    def test_processor_pulse_generation_with_bad_tokens(self):
        NUM_SAMPLES = 100
        INP_PULSE_LENGTH = 10

        # prevent any pulses from being generated for the first 10 steps
        # the disrupt any pulses at step 15
        duration = np.zeros(self.NUM_PROCESSORS) + 10
        threshold = np.arange(1, self.NUM_PROCESSORS+1)

        # set parameters
        for i in range(self.NUM_PROCESSORS):
            pp = self.proc[i].parameters
            pp.goodTokenThreshold = threshold[i]
            pp.badTokenThreshold = 1
            pp.duration = duration[i]
        
        # set inputs, record states and outputs
        # generate constant good token inputs
        my_good_tokens_in = np.ones((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        my_bad_tokens_in = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)

        bad_token_times = np.arange(5, 100, 15)
        bad_token_times[0] = 0
        my_bad_tokens_in[bad_token_times,:] = 1

        # give each token a lifetime of DELAY
        PyTTT.set_expiration(my_good_tokens_in, INP_PULSE_LENGTH, circular=False)
        PyTTT.set_expiration(my_bad_tokens_in, INP_PULSE_LENGTH, circular=False)

        # each processor should fire as soon as the inhibition is removed,
        # and each pulse be interrupted by the next bad token
        should_start = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        should_stop = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        should_start[bad_token_times[:-1]+11, :] = 1
        should_stop[bad_token_times[1:], :] = 1

        # run processors, iterating over the inputs
        did_start = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        did_stop = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        for step,(start_token, stop_token) in enumerate(self.proc.run(my_good_tokens_in, my_bad_tokens_in)):
            did_start[step,:] = start_token
            did_stop[step,:] = stop_token

        # check that the pulse generation works correctly
        self.assertSequenceEqual(did_start.tolist(), should_start.tolist())
        self.assertSequenceEqual(did_stop.tolist(), should_stop.tolist())

    def test_recurrent_tokens(self):
        NUM_SAMPLES = 100

        # connect every processor to the next (excitatory)
        self.proc.W_good[:,:] = np.eye(self.NUM_PROCESSORS, k=-1)
        self.proc.W_good[0,-1] = 1

        # block the second nearest processor
        self.proc.W_bad[:,:] = np.eye(self.NUM_PROCESSORS)#, k=1)
        # self.proc.W_bad[-1,-1] = 1

        # set parameters
        self.proc.goodTokenThreshold[:] = 1
        self.proc.badTokenThreshold[:] = 1
        self.proc.duration[:] = 10

        # kick off a cascade with an initial pulse
        my_good_tokens_in = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        my_bad_tokens_in = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        my_good_tokens_in[0,0] = 1
        my_good_tokens_in[1,0] = -1
        
        # each processor should fire as soon as the inhibition is removed,
        # and each pulse be interrupted by the next bad token
        should_start = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        should_stop = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        for i in range(NUM_SAMPLES):
            should_start[i, i % self.NUM_PROCESSORS] = 1
            if i+1 < NUM_SAMPLES:
                should_stop[i+1, i % self.NUM_PROCESSORS] = 1

        # run processors, iterating over the inputs
        did_start = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        did_stop = np.zeros((NUM_SAMPLES, self.NUM_PROCESSORS), dtype=int)
        for step,(start_token, stop_token) in enumerate(self.proc.run(my_good_tokens_in, my_bad_tokens_in)):
            did_start[step,:] = start_token
            did_stop[step,:] = stop_token

        # check that the pulse generation works correctly
        self.assertSequenceEqual(did_start.tolist(), should_start.tolist())
        self.assertSequenceEqual(did_stop.tolist(), should_stop.tolist())


if __name__ == '__main__':
    unittest.main()