from hiive import mdptoolbox
from hiive.mdptoolbox.mdp import MDP, _MSG_STOP_MAX_ITER, _MSG_STOP_UNCHANGING_POLICY, _printVerbosity, \
    _MSG_STOP_EPSILON_OPTIMAL_VALUE, _computeDimensions

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

import hiive.mdptoolbox.util as _util


class QLearningCustom(MDP):
    """A discounted MDP solved using the Q learning algorithm.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    gamma : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    n_iter : int, optional
        Number of iterations to execute. This is ignored unless it is an
        integer greater than the default value. Defaut: 10,000.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    ---------------
    Q : array
        learned Q matrix (SxA)
    V : tuple
        learned value function (S).
    policy : tuple
        learned optimal policy (S).
    mean_discrepancy : array
        Vector of V discrepancy mean over 100 iterations. Then the length of
        this vector for the default value of N is 100 (N/100).

    Examples
    ---------
    >>> # These examples are reproducible only if random seed is set to 0 in
    >>> # both the random and numpy.random modules.
    >>> import numpy as np
    >>> import hiive.mdptoolbox, hiive.mdptoolbox.example
    >>> np.random.seed(0)
    >>> P, R = mdptoolbox.example.forest()
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
    >>> ql.run()
    >>> ql.Q
    array([[ 11.198909  ,  10.34652034],
           [ 10.74229967,  11.74105792],
           [  2.86980001,  12.25973286]])
    >>> expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (0, 1, 1)

    >>> import hiive.mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> np.random.seed(0)
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.9)
    >>> ql.run()
    >>> ql.Q
    array([[ 33.33010866,  40.82109565],
           [ 34.37431041,  29.67236845]])
    >>> expected = (40.82109564847122, 34.37431040682546)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (1, 0)

    """

    def __init__(self, transitions, reward, gamma,
                 alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99,
                 n_iter=10000, skip_check=False, iter_callback=None,
                 run_stat_frequency=None, episode_length=100, td_error_threshold=0.001, overall_stat_freq=100):
        # Initialise a Q-learning MDP.

        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        if not skip_check:
            # We don't want to send this to MDP because _computePR should not
            #  be run on it, so check that it defines an MDP
            _util.check(transitions, reward)

        # Store P, S, and A
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)

        self.R = reward

        self.alpha = _np.clip(alpha, 0., 1.)
        self.alpha_start = self.alpha
        self.alpha_decay = _np.clip(alpha_decay, 0., 1.)
        self.alpha_min = _np.clip(alpha_min, 0., 1.)
        self.gamma = _np.clip(gamma, 0., 1.)
        self.epsilon = _np.clip(epsilon, 0., 1.)
        self.epsilon_start = self.epsilon
        self.epsilon_decay = _np.clip(epsilon_decay, 0., 1.)
        self.epsilon_min = _np.clip(epsilon_min, 0., 1.)
        self.episode_length = episode_length

        # Initialisations
        self.Q = _np.zeros((self.S, self.A))

        self.run_stats = []
        self.error_mean = []
        self.v_mean = []
        self.p_cumulative = []
        self.iter_callback = iter_callback
        self.S_freq = _np.zeros((self.S, self.A))
        self.run_stat_frequency = max(1, self.max_iter // 10000) if run_stat_frequency is None else run_stat_frequency

        self.td_error_threshold = td_error_threshold

        self.overall_stat_freq = overall_stat_freq
        self.stat_error_mean = []
        self.stat_error_max = []
        self.stat_iters = []

    def run(self):

        stat_error_cum = []

        # Run the Q-learning algorithm.
        error_cumulative = []
        self.run_stats = []
        self.error_mean = []

        v_cumulative = []
        self.v_mean = []

        self.p_cumulative = []

        self.time = _time.time()

        # initial state choice
        s = _np.random.randint(0, self.S)
        reset_s = False
        run_stats = []
        for n in range(1, self.max_iter + 1):
            take_overall_stat = n % self.overall_stat_freq == 0 or n == self.max_iter

            take_run_stat = n % self.run_stat_frequency == 0 or n == self.max_iter

            # Reinitialisation of trajectories every 100 transitions
            if (self.iter_callback is None and (n % self.episode_length) == 0) or reset_s:
                s = _np.random.randint(0, self.S)

            # Action choice : greedy with increasing probability
            # The agent takes random actions for probability ε and greedy action for probability (1-ε).
            pn = _np.random.random()
            if pn < self.epsilon:
                a = _np.random.randint(0, self.A)
            else:
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()

            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = _np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            # Q[s, a] = Q[s, a] + alpha*(R + gamma*Max[Q(s’, A)] - Q[s, a])
            # Updating the value of Q
            dQ = self.alpha * (r + self.gamma * self.Q[s_new, :].max() - self.Q[s, a])
            self.Q[s, a] = self.Q[s, a] + dQ

            # Computing means all over maximal Q variations values
            error = _np.absolute(dQ)

            # compute the value function and the policy
            v = self.Q.max(axis=1)
            self.V = v
            p = self.Q.argmax(axis=1)
            self.policy = p

            self.S_freq[s, a] += 1
            run_stats.append(self._build_run_stat(i=n, s=s, a=a, r=r, p=p, v=v, error=error))

            stat_error_cum.append(error)

            if take_overall_stat:
                self.stat_error_mean.append(_np.mean(stat_error_cum))
                self.stat_error_max.append(_np.max(stat_error_cum))
                stat_error_cum = []
                self.stat_iters.append(n)

            if take_run_stat or len(self.stat_error_mean) > 0 and self.stat_error_mean[-1] < self.td_error_threshold:
                error_cumulative.append(error)

                if len(error_cumulative) == 100:
                    self.error_mean.append(_np.mean(error_cumulative))
                    error_cumulative = []

                v_cumulative.append(v)

                if len(v_cumulative) == 100:
                    self.v_mean.append(_np.mean(v_cumulative, axis=1))
                    v_cumulative = []

                if len(self.p_cumulative) == 0 or not _np.array_equal(self.policy, self.p_cumulative[-1][1]):
                    self.p_cumulative.append((n, self.policy.copy()))
                """
                Rewards,errors time at each iteration I think
                But that’s for all of them and steps per episode?

                Alpha decay and min ?
                And alpha and epsilon at each iteration?
                """
                self.run_stats.append(run_stats[-1])
                run_stats = []

            if self.iter_callback is not None:
                reset_s = self.iter_callback(s, a, s_new)

            # current state is updated
            s = s_new

            self.alpha *= self.alpha_decay
            if self.alpha < self.alpha_min:
                self.alpha = self.alpha_min

            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

            if len(self.stat_error_mean) > 0 and self.stat_error_mean[-1] < self.td_error_threshold:
                break

        self._endRun()
        # add stragglers
        if len(v_cumulative) > 0:
            self.v_mean.append(_np.mean(v_cumulative, axis=1))
        if len(error_cumulative) > 0:
            self.error_mean.append(_np.mean(error_cumulative))
        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats

        return self.run_stats

    def _build_run_stat(self, i, a, error, p, r, s, v):
        run_stat = {
            'State': s,
            'Action': a,
            'Reward': r,
            'Error': error,
            'Time': _time.time() - self.time,
            'Alpha': self.alpha,
            'Epsilon': self.epsilon,
            'Gamma': self.gamma,
            'V[0]': v[0],
            'Max V': _np.max(v),
            'Mean V': _np.mean(v),
            'Iteration': i,
            # 'Value': v.copy(),
            # 'Policy': p.copy()
        }
        return run_stat
