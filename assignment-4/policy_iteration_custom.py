from hiive import mdptoolbox
from hiive.mdptoolbox.mdp import MDP, _MSG_STOP_MAX_ITER, _MSG_STOP_UNCHANGING_POLICY, _printVerbosity, \
    _MSG_STOP_EPSILON_OPTIMAL_VALUE

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

import hiive.mdptoolbox.util as _util


class PolicyIterationCustom(MDP):
    """A discounted MDP solved using the policy iteration algorithm.

    Arguments
    ---------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    gamma : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    policy0 : array, optional
        Starting policy.
    max_iter : int, optional
        Maximum number of iterations. See the documentation for the ``MDP``
        class for details. Default is 1000.
    eval_type : int or string, optional
        Type of function used to evaluate policy. 0 or "matrix" to solve as a
        set of linear equations. 1 or "iterative" to solve iteratively.
        Default: 0.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    ---------------
    V : tuple
        value function
    policy : tuple
        optimal policy
    iter : int
        number of done iterations
    time : float
        used CPU time

    Notes
    -----
    In verbose mode, at each iteration, displays the number
    of differents actions between policy n-1 and n

    Examples
    --------
    >>> import hiive.mdptoolbox, hiive.mdptoolbox.example
    >>> P, R = mdptoolbox.example.rand(10, 3)
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    >>> pi.run()

    >>> P, R = mdptoolbox.example.forest()
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    >>> pi.run()
    >>> expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
    >>> all(expected[k] - pi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> pi.policy
    (0, 0, 0)
    """

    def __init__(self, transitions, reward, gamma, policy0=None,
                 max_iter=1000, eval_type=1, skip_check=False,
                 run_stat_frequency=None):
        # Initialise a policy iteration MDP.
        #
        # Set up the MDP, but don't need to worry about epsilon values
        MDP.__init__(self, transitions, reward, gamma, None, max_iter,
                     skip_check=skip_check)
        # Check if the user has supplied an initial policy. If not make one.
        self.run_stats = None
        if policy0 is None:
            # Initialise the policy to the one which maximises the expected
            # immediate reward
            null = _np.zeros(self.S)
            self.policy, null = self._bellmanOperator(null)
            del null
        else:
            # Use the policy that the user supplied
            # Make sure it is a numpy array
            policy0 = _np.array(policy0)
            # Make sure the policy is the right size and shape
            assert policy0.shape in ((self.S,), (self.S, 1), (1, self.S)), \
                "'policy0' must a vector with length S."
            # reshape the policy to be a vector
            policy0 = policy0.reshape(self.S)
            # The policy can only contain integers between 0 and S-1
            msg = "'policy0' must be a vector of integers between 0 and S-1."
            assert not _np.mod(policy0, 1).any(), msg
            assert (policy0 >= 0).all(), msg
            assert (policy0 < self.S).all(), msg
            self.policy = policy0
        # set the initial values to zero
        self.V = _np.zeros(self.S)
        self.error_mean = []
        self.v_mean = []
        self.p_cumulative = []
        self.run_stat_frequency = max(1, max_iter // 10000) if run_stat_frequency is None else run_stat_frequency

        # Do some setup depending on the evaluation type
        if eval_type in (0, "matrix"):
            self.eval_type = "matrix"
        elif eval_type in (1, "iterative"):
            self.eval_type = "iterative"
        else:
            raise ValueError("'eval_type' should be '0' for matrix evaluation "
                             "or '1' for iterative evaluation. The strings "
                             "'matrix' and 'iterative' can also be used.")

    def _computePpolicyPRpolicy(self):
        # Compute the transition matrix and the reward matrix for a policy.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA)  = transition matrix
        #     P could be an array with 3 dimensions or a cell array (1xA),
        #     each cell containing a matrix (SxS) possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #     R could be an array with 3 dimensions (SxSxA) or
        #     a cell array (1xA), each cell containing a sparse matrix (SxS) or
        #     a 2D array(SxA) possibly sparse
        # policy(S) = a policy
        #
        # Evaluation
        # ----------
        # Ppolicy(SxS)  = transition matrix for policy
        # PRpolicy(S)   = reward matrix for policy
        #
        Ppolicy = _np.empty((self.S, self.S))
        Rpolicy = _np.zeros(self.S)
        for aa in range(self.A):  # avoid looping over S
            # the rows that use action a.
            ind = (self.policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign this
            if ind.size > 0:
                try:
                    Ppolicy[ind, :] = self.P[aa][ind, :]
                except ValueError:
                    Ppolicy[ind, :] = self.P[aa][ind, :].todense()
                # PR = self._computePR() # an apparently uneeded line, and
                # perhaps harmful in this implementation c.f.
                # mdp_computePpolicyPRpolicy.m
                Rpolicy[ind] = self.R[aa][ind]
        # self.R cannot be sparse with the code in its current condition, but
        # it should be possible in the future. Also, if R is so big that its
        # a good idea to use a sparse matrix for it, then converting PRpolicy
        # from a dense to sparse matrix doesn't seem very memory efficient
        if type(self.R) is _sp.csr_matrix:
            Rpolicy = _sp.csr_matrix(Rpolicy)
        # self.Ppolicy = Ppolicy
        # self.Rpolicy = Rpolicy
        return Ppolicy, Rpolicy

    def _evalPolicyIterative(self, V0=0, epsilon=0.0001, max_iter=10000):
        # Evaluate a policy using iteration.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA)  = transition matrix
        #    P could be an array with 3 dimensions or
        #    a cell array (1xS), each cell containing a matrix possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #    R could be an array with 3 dimensions (SxSxA) or
        #    a cell array (1xA), each cell containing a sparse matrix (SxS) or
        #    a 2D array(SxA) possibly sparse
        # discount  = discount rate in ]0; 1[
        # policy(S) = a policy
        # V0(S)     = starting value function, optional (default : zeros(S,1))
        # epsilon   = epsilon-optimal policy search, upper than 0,
        #    optional (default : 0.0001)
        # max_iter  = maximum number of iteration to be done, upper than 0,
        #    optional (default : 10000)
        #
        # Evaluation
        # ----------
        # Vpolicy(S) = value function, associated to a specific policy
        #
        # Notes
        # -----
        # In verbose mode, at each iteration, displays the condition which
        # stopped iterations: epsilon-optimum value function found or maximum
        # number of iterations reached.
        #
        try:
            assert V0.shape in ((self.S,), (self.S, 1), (1, self.S)), \
                "'V0' must be a vector of length S."
            policy_V = _np.array(V0).reshape(self.S)
        except AttributeError:
            if V0 == 0:
                policy_V = _np.zeros(self.S)
            else:
                policy_V = _np.array(V0).reshape(self.S)

        policy_P, policy_R = self._computePpolicyPRpolicy()

        if self.verbose:
            _printVerbosity("Iteration", "V variation")

        itr = 0
        done = False

        while not done:
            itr += 1

            Vprev = policy_V
            policy_V = policy_R + self.gamma * policy_P.dot(Vprev)

            variation = _np.absolute(policy_V - Vprev).max()
            if self.verbose:
                _printVerbosity(itr, variation)

            # ensure |Vn - Vpolicy| < epsilon
            if variation < ((1 - self.gamma) / self.gamma) * epsilon:
                done = True
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_VALUE)
            elif itr == max_iter:
                done = True
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)

        self.V = policy_V
        return policy_V, policy_R, itr

    def _evalPolicyMatrix(self):
        # Evaluate the value function of the policy using linear equations.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA) = transition matrix
        #      P could be an array with 3 dimensions or a cell array (1xA),
        #      each cell containing a matrix (SxS) possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #      R could be an array with 3 dimensions (SxSxA) or
        #      a cell array (1xA), each cell containing a sparse matrix (SxS)
        #      or a 2D array(SxA) possibly sparse
        # discount = discount rate in ]0; 1[
        # policy(S) = a policy
        #
        # Evaluation
        # ----------
        # Vpolicy(S) = value function of the policy
        #
        Ppolicy, Rpolicy = self._computePpolicyPRpolicy()
        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
        policy_V = _np.linalg.solve((_sp.eye(self.S, self.S) - self.gamma * Ppolicy), Rpolicy)
        self.V = policy_V
        return policy_V, Rpolicy, None

    def _build_run_stat(self, i, s, a, r, p, v, error, changes):
        run_stat = {
            'State': s,
            'Action': a,
            'Reward': r,
            'Error': error,
            'Time': _time.time() - self.time,
            'V[0]': v[0],
            'Max V': _np.max(v),
            'Mean V': _np.mean(v),
            'Iteration': i,
            'Changes': changes,
            # 'Value': v.copy(),
            # 'Policy': p.copy()
        }
        return run_stat

    def run(self):
        # Run the policy iteration algorithm.
        self._startRun()
        self.run_stats = []

        self.error_mean = []
        error_cumulative = []

        self.v_mean = []
        v_cumulative = []

        self.p_cumulative = []
        run_stats = []
        while True:
            self.iter += 1
            take_run_stat = self.iter % self.run_stat_frequency == 0 or self.iter == self.max_iter
            # these _evalPolicy* functions will update the classes value
            # attribute
            policy_V, policy_R, itr = (self._evalPolicyMatrix()
                                       if self.eval_type == 'matrix'
                                       else self._evalPolicyIterative())

            if take_run_stat:
                v_cumulative.append(policy_V)
                if len(v_cumulative) == 100:
                    self.v_mean.append(_np.mean(v_cumulative, axis=1))
                    v_cumulative = []
                if len(self.p_cumulative) == 0 or not _np.array_equal(self.policy, self.p_cumulative[-1][1]):
                    self.p_cumulative.append((self.iter, self.policy.copy()))



            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, next_v = self._bellmanOperator()
            # calculate in how many places does the old policy disagree with
            # the new policy
            nd = (policy_next != self.policy).sum()
            error = _np.absolute(next_v - policy_V).max()
            run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(policy_V),
                                                  p=policy_next, v=policy_V, error=error, changes=nd))

            if take_run_stat:
                error_cumulative.append(error)
                if len(error_cumulative) == 100:
                    self.error_mean.append(_np.mean(error_cumulative))
                    error_cumulative = []
                self.run_stats.append(run_stats[-1])
                run_stats = []
            del next_v

            # if verbose then continue printing a table
            if self.verbose:
                _printVerbosity(self.iter, nd)
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop

            # Error, rewards, and time for every iteration and number of PI steps which might be specific to my setup
            if nd == 0:
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break
            else:
                self.policy = policy_next

        self._endRun()
        # add stragglers
        if len(v_cumulative) > 0:
            self.v_mean.append(_np.mean(v_cumulative, axis=1))
        if len(error_cumulative) > 0:
            self.error_mean.append(_np.mean(error_cumulative))
        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats
        return self.run_stats
