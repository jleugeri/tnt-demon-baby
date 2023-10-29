import numpy as np
from dataclasses import dataclass, field
import numba
import typing

PyTTT = typing.NewType("PyTTT", object)

class SerializeToDict(object):
    dontSerializeProps = []

    def serialize_to_dict(self) -> typing.Dict[str,object]:
        ret= {}
        for prop in dir(self):
            if prop == "dontSerializeProps" or prop in self.__class__.dontSerializeProps or prop.startswith("__") or callable(getattr(type(self), prop, None)):
                continue
            val = getattr(self, prop)

            if hasattr(val, "serialize_to_dict"):
                val = val.serialize_to_dict()

            ret[prop] = val

        return ret

@dataclass(init=False, repr=False)
class ProcessorInfo(SerializeToDict):
    """Utility class to simplify access to the parameters and state of each processor by name"""
    dontSerializeProps=["procs"]

    @dataclass
    class ProcessorState(SerializeToDict):
        dontSerializeProps=["procs", "key"]
        procs: PyTTT
        key: int

        @property
        def isOn(self) -> bool:
            return self.procs.isOn[self.key]
        
        @isOn.setter
        def isOn(self, val) -> ():
            self.procs.isOn[self.key] = val
        

        @property
        def goodTokens(self) -> int:
            return self.procs.goodTokens[self.key]
        
        @goodTokens.setter
        def goodTokens(self, val) -> ():
            self.procs.goodTokens[self.key] = val
        

        @property
        def badTokens(self) -> int:
            return self.procs.badTokens[self.key]
        
        @badTokens.setter
        def badTokens(self, val) -> ():
            self.procs.badTokens[self.key] = val
        

        @property
        def timeout(self) -> int:
            return self.procs.timeout[self.key]
    
        @timeout.setter
        def timeout(self, val) -> ():
            self.procs.timeout[self.key] = val

        def __repr__(self):
            return "\n    isOn: {}\n    goodTokens: {}\n    badTokens: {}\n    timeout: {}\n".format(self.isOn, self.goodTokens, self.badTokens, self.timeout)


    
    @dataclass
    class ProcessorIO(SerializeToDict):
        dontSerializeProps=["procs", "key"]
        procs: object
        key: int

        @property
        def goodTokensIn(self) -> int:
            return self.procs.goodTokensIn[self.key]
        
        @goodTokensIn.setter
        def goodTokensIn(self, val) -> ():
            self.procs.goodTokensIn[self.key] = val
        

        @property
        def badTokensIn(self) -> int:
            return self.procs.badTokensIn[self.key]
        
        @badTokensIn.setter
        def badTokensIn(self, val) -> ():
            self.procs.badTokensIn[self.key] = val
        

        @property
        def startTokens(self) -> bool:
            return self.procs.startTokens[self.key]
        
        @startTokens.setter
        def startTokens(self, val) -> ():
            self.procs.startTokens[self.key] = val

        @property
        def extendTokens(self) -> bool:
            return self.procs.extendTokens[self.key]
        
        @extendTokens.setter
        def extendTokens(self, val) -> ():
            self.procs.extendTokens[self.key] = val
        

        @property
        def stopTokens(self) -> bool:
            return self.procs.stopTokens[self.key]
    
        @stopTokens.setter
        def stopTokens(self, val) -> ():
            self.procs.stopTokens[self.key] = val

        def __repr__(self):
            return "\n    goodTokensIn: {}\n    badTokensIn: {}\n    startTokens: {}\n    extendTokens: {}\n    stopTokens: {}\n".format(self.goodTokensIn, self.badTokensIn, self.startTokens, self.extendTokens, self.stopTokens)
    

    
    @dataclass
    class ProcessorParameters(SerializeToDict):
        dontSerializeProps=["procs", "key"]
        procs: PyTTT
        key: int

        @property
        def goodTokenThreshold(self) -> int:
            return self.procs.goodTokenThreshold[self.key]
        
        @goodTokenThreshold.setter
        def goodTokenThreshold(self, val) -> ():
            self.procs.goodTokenThreshold[self.key] = val
        

        @property
        def badTokenThreshold(self) -> int:
            return self.procs.badTokenThreshold[self.key]
        
        @badTokenThreshold.setter
        def badTokenThreshold(self, val) -> ():
            self.procs.badTokenThreshold[self.key] = val
        

        @property
        def duration(self) -> int:
            return self.procs.duration[self.key]

        @duration.setter
        def duration(self, val) -> ():
            self.procs.duration[self.key] = val

        def __repr__(self):
            return "\n    goodTokenThreshold: {}\n    badTokenThreshold: {}\n    duration: {}\n".format(self.goodTokenThreshold, self.badTokenThreshold, self.duration)


    @dataclass
    class ProcessorConnectionsIn(SerializeToDict):
        dontSerializeProps=["procs", "key"]
        procs: PyTTT
        key: int

        @property
        def W_good(self) -> np.ndarray:
            return self.procs.W_good[self.key,:]
        
        @W_good.setter
        def W_good(self, val) -> ():
            self.procs.W_good[self.key,:] = val
        

        @property
        def W_bad(self) -> np.ndarray:
            return self.procs.W_bad[self.key,:]
        
        @W_bad.setter
        def W_bad(self, val) -> ():
            self.procs.W_bad[self.key,:] = val

        def __repr__(self):
            return "\n    W_good: {{{}}}\n    W_bad: {{{}}}\n".format(
                ",".join(
                    ("{}: {}".format(i, self.W_good[i]) for i in np.nonzero(self.W_good)[0])
                ),
                ",".join(
                    ("{}: {}".format(i, self.W_bad[i]) for i in np.nonzero(self.W_bad)[0])
                ),
            )


    @dataclass
    class ProcessorConnectionsOut(SerializeToDict):
        dontSerializeProps=["procs", "key"]
        procs: PyTTT
        key: int

        @property
        def W_good(self) -> np.ndarray:
            return self.procs.W_good[:, self.key]
        
        @W_good.setter
        def W_good(self, val) -> ():
            self.procs.W_good[:, self.key] = val
        

        @property
        def W_bad(self) -> np.ndarray:
            return self.procs.W_bad[:, self.key]
        
        @W_bad.setter
        def W_bad(self, val) -> ():
            self.procs.W_bad[:, self.key] = val

        def __repr__(self):
            return "\n    W_good: {{{}}}\n    W_bad: {{{}}}\n".format(
                ",".join(
                    ("{}: {}".format(i, self.W_good[i]) for i in np.nonzero(self.W_good)[0])
                ),
                ",".join(
                    ("{}: {}".format(i, self.W_bad[i]) for i in np.nonzero(self.W_bad)[0])
                ),
            )

    def __repr__(self):
        return "Processor #{}: \n  state: {}\n  io: {}\n  parameters: {}\n  connectionsIn: {}\n  connectionsOut: {}".format(self.key, self.state, self.io, self.parameters, self.connectionsIn, self.connectionsOut)
    
    procs: PyTTT
    key: int
    state: ProcessorState
    io: ProcessorIO
    parameters: ProcessorParameters
    connectionsOut: ProcessorConnectionsOut
    connectionsIn: ProcessorConnectionsIn

    def __init__(self, procs, key):
        self.procs = procs
        self.key = key

        self.state = ProcessorInfo.ProcessorState(self.procs, self.key)
        self.io = ProcessorInfo.ProcessorIO(self.procs, self.key)
        self.parameters = ProcessorInfo.ProcessorParameters(self.procs, self.key)
        self.connectionsIn = ProcessorInfo.ProcessorConnectionsIn(self.procs, self.key)
        self.connectionsOut = ProcessorInfo.ProcessorConnectionsOut(self.procs, self.key)

class PyTTT():
    def __init__(self, goodThreshold, badThreshold, W_good, W_bad, duration, initialGoodTokens=None, initialBadTokens=None, initialDuration=None):
        """Create a new instance of the TickTockToken processor.

        Parameters
        ----------
        goodThreshold : array_like
            The good token threshold for each processor.
        badThreshold : array_like
            The bad token threshold for each processor.
        W_good : array_like
            The internal good token weight matrix, i.e. the number of good tokens that each processor sends to each other processor.
        W_bad : array_like
            The internal bad token weight matrix, i.e. the number of bad tokens that each processor sends to each other processor.
        duration : array_like
            The duration of the tokens generated by each processor.
            Note that `duration' sets the time steps between token start and token stop, 
            i.e. if duration=d, and the token is generated in step t, then:
              - a start_token signal will be emitted in step t
              - the internal timer will start ticking down d steps from step t+1, reaching 0 in step t+1+d
              - a stop_token signal will be emitted in step t+1+d
            This implies that for duration 0, the processor can at most fire a new token every other step,
            i.e. it cannot generate a start and a stop token signal in the same step.
        """
        self.NUM_PROCESSORS = len(goodThreshold)
        
        assert len(badThreshold) == self.NUM_PROCESSORS, "good_threshold and bad_threshold must have the same length"
        assert W_good.shape == W_bad.shape == (self.NUM_PROCESSORS, self.NUM_PROCESSORS), "W_good and W_bad must be square matrices of size ({},{})".format(self.NUM_PROCESSORS, self.NUM_PROCESSORS)

        # keep track of the internal states of the processors
        self.goodTokens = np.zeros((self.NUM_PROCESSORS,), dtype=np.int32)
        if initialGoodTokens is not None:
            self.goodTokens[:] = initialGoodTokens 
        self.badTokens = np.zeros((self.NUM_PROCESSORS,), dtype=np.int32)
        if initialBadTokens is not None:
            self.badTokens[:] = initialBadTokens 
        self.timeout = np.zeros((self.NUM_PROCESSORS,), dtype=np.uint32)
        if initialDuration is not None:
            self.timeout[:] = initialDuration
        self.isOn = self.timeout > 0

        # the outputs of each processor in the current time-step
        self.startExtendStopTokens = np.zeros((3,self.NUM_PROCESSORS), dtype=bool)
        self.startTokens = self.startExtendStopTokens.view()[0,:]
        self.extendTokens = self.startExtendStopTokens.view()[1,:]
        self.stopTokens = self.startExtendStopTokens.view()[2,:]

        # the parameters of each processor: thresholds, durations and weights
        self.goodTokenThreshold = np.zeros((self.NUM_PROCESSORS,), dtype=np.int32)
        self.goodTokenThreshold[:] = goodThreshold
        self.badTokenThreshold = np.zeros((self.NUM_PROCESSORS,), dtype=np.int32)
        self.badTokenThreshold[:] = badThreshold
        self.duration = np.zeros((self.NUM_PROCESSORS,), dtype=np.uint32)
        self.duration[:] = duration
        
        self.W_good = W_good
        self.W_bad = W_bad


    """
    def step_processors_numpy(self):
        # accumulate incoming tokens
        self.good_tokens_state += self.good_tokens_in
        self.bad_tokens_state += self.bad_tokens_in

        # check firing conditions
        self.may_turn_on[:] = self.good_tokens_state >= self.good_tokens_threshold
        self.may_turn_off[:] = self.bad_tokens_state >= self.bad_tokens_threshold

        np.logical_not(self.may_turn_off, out=self.may_turn_off_n)
        np.logical_and(self.may_turn_on, self.may_turn_off_n, out=self.tmp)
        # compute start_token[i] = !isOn[i] & may_turn_on[i] &  may_turn_off & 
        np.logical_and(self.isOn_n, self.tmp, out=self.start_token)
        np.logical_and(self.isOn, self.may_turn_off, out=self.stop_token)

        # update the state
        self.isOn[self.start_token] = True
        self.isOn[self.stop_token] = False
    """

    @staticmethod
    @numba.jit(nopython=True)
    def _check_state(isOn, good, bad, good_thresh, bad_thresh, timeout, token_start, token_stop, token_extend):
        n = len(token_start)
        for i in range(n):
            may_turn_on = good[i] >= good_thresh[i]
            may_turn_off = bad[i] > bad_thresh[i]
            is_expired = timeout[i] == 0
            token_start[i] = ~isOn[i] & may_turn_on & ~may_turn_off
            token_extend[i] = isOn[i] & is_expired & may_turn_on & ~may_turn_off
            token_stop[i] = isOn[i] & (may_turn_off | (is_expired & ~token_extend[i])) 

    def stimulate_processors(self, good_tokens_in, bad_tokens_in):
        # accumulate incoming tokens
        self.goodTokens += good_tokens_in
        self.badTokens += bad_tokens_in

    def step_processors(self):
        """Compute the processors' state updates from the processors' inputs {good,bad}_tokens_in, and produce the outputs {start,stop}_token.
        """
        
        # check firing conditions
        self._check_state(
            self.isOn, 
            self.goodTokens, self.badTokens, 
            self.goodTokenThreshold, self.badTokenThreshold,
            self.timeout,
            self.startTokens,
            self.stopTokens,
            self.extendTokens
        )

        # count down timers
        self.timeout -= self.isOn

        # update the states
        self.isOn[self.startTokens] = True
        self.timeout[self.startTokens] = self.duration[self.startTokens]

        self.timeout[self.extendTokens] = self.duration[self.extendTokens]

        self.isOn[self.stopTokens] = False
        self.timeout[self.stopTokens] = 0


    def step_NoC(self):
        """Compute the NoC update from the processors' outputs {start,stop}_token, and produce the inputs {good,bad}_tokens_in for the next cycle."""
        delta_tokens = (self.startTokens.astype(int) - self.stopTokens.astype(int))
        self.stimulate_processors(self.W_good @ delta_tokens, self.W_bad @ delta_tokens)

    def run(self, inp_good_tokens_iter, inp_bad_tokens_iter):
        inp_good_tokens_iter = iter(inp_good_tokens_iter)
        inp_bad_tokens_iter = iter(inp_bad_tokens_iter)

        for tok_good,tok_bad in zip(inp_good_tokens_iter, inp_bad_tokens_iter):
            # reset inputs, add external inputs
            self.stimulate_processors(tok_good, tok_bad)

            # add recurrent inputs
            self.step_NoC()

            # update each processor
            self.step_processors()

            # yield the outputs
            yield (self.startTokens, self.stopTokens)


    @staticmethod
    def set_expiration(token_starts: np.ndarray, delay: typing.Union[int, np.ndarray], circular=False):
        """For the array of token_starts, subtracts the corresponding token count after the specified delays (in place!)."""
        if isinstance(delay, np.ndarray):
            assert len(delay) == token_starts.shape[1], "Number of delays ({}) must equal number of processors ({})".format(len(delay), token_starts.shape[1])
            # recurse for each different delay
            for t,d in zip(token_starts.T, delay):
                PyTTT.set_expiration(t, d, circular=circular)
        elif isinstance(delay, int) or isinstance(delay, np.integer):
            if circular:
                token_starts -= np.roll(token_starts, delay+1, axis=0)
            else:
                if(delay < token_starts.shape[0]-1):
                    token_starts[delay+1:,...] -= token_starts[:token_starts.shape[0]-1-delay,...]
        else:
            raise TypeError("delay must be either an integer or an array of integers, not {}".format(type(delay)))

    def __getitem__(self, key: int):
        assert 0 <= key <= self.NUM_PROCESSORS, KeyError("No processor with index {}".format(key))
        return ProcessorInfo(self, key)

if __name__ == "__main__":

    # set up parameters
    NUM_PROCESSORS = 100
    NUM_SAMPLES = 100
    DELAY = 10

    duration = 10
    good_threshold = np.zeros(NUM_PROCESSORS)
    bad_threshold = np.ones(NUM_PROCESSORS)
    W_good = np.random.poisson(1,(NUM_PROCESSORS,NUM_PROCESSORS))
    W_bad = np.random.poisson(1,(NUM_PROCESSORS,NUM_PROCESSORS))

    proc = PyTTT(good_threshold, bad_threshold, W_good, W_bad, duration)
