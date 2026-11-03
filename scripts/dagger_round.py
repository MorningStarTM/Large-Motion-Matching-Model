"""
One DAgger round (plan section 5.3 step 3).

Run MMM, query the KNN teacher at the states MMM reaches, add those pairs to the
dataset, fine tune again. Repeat a few rounds by running this script again.

Kept as its own entry point because each round should be evaluated before the
next one is started: a round that makes things worse must be visible
immediately.
"""
