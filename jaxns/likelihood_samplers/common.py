from collections import namedtuple

SamplingResults = namedtuple('SamplingResults',
                                  ['key', 'num_likelihood_evaluations', 'u_new', 'log_L_new'])