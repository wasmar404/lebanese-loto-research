"""Prediction models.

Every model produces an (N, 42) matrix ``P`` where ``P[t]`` is the forecast probability that
each number 1..42 appears among the six main numbers of draw ``t``, computed **only from
draws 0..t-1**.  A ticket is the six numbers with the highest probability.
"""
