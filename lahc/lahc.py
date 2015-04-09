from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import copy
import sys
import signal


class LAHC(object):
    """Performs Least Acceptance Hill Climbing algorithm

        Parameters
        lfa : length of fitness array, proportional to cpu time and sol quality
        initial_state : Starting point in the search space
    """

    copy_strategy = 'deepcopy'
    user_exit = False

    def __init__(self, initial_state, lfa=1000, steps=int(100e6)):
        self.initial_state = initial_state
        self.lfa = lfa
        self.steps = steps
        self.state = self.copy_state(initial_state)
        c = self.cost()
        self.f = [c for i in range(self.lfa)]
        signal.signal(signal.SIGINT, self.set_user_exit)

    def set_user_exit(self, signum, frame):
        """Raises the user_exit flag, further iterations are stopped
        """
        self.user_exit = True

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy : use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()

    def optimize(self):
        """Optimizes the state through LAHC meta-heuristic

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0

        # Note initial state
        lastAccepted = self.copy_state(self.state)
        bestState = self.copy_state(self.state)
        C = self.cost()
        lastAcceptedCost = C
        bestCost = C

        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            step += 1
            self.move()
            C = self.cost()  # Candidate cost
            v = step % self.lfa
            if C < self.f[v] or C < lastAcceptedCost:
                # Accept solution
                lastAccepted = self.copy_state(self.state)
                lastAcceptedCost = C
                if C < bestCost:
                    bestCost = C
                    bestState = self.copy_state(self.state)
                    sys.stdout.write("\rBest Cost found: %12.2f, Total progress: %d%%\n" % (bestCost, int(step/self.steps*100) ) )
                    sys.stdout.flush()
            else:
                # Reject solution, reset to lastAccepted
                self.state = self.copy_state(lastAccepted)
                C = lastAcceptedCost # TODO: is this needed?

            self.f[v] = C

        # Return best state and energy
        return bestState, bestCost
