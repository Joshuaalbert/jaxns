In recent years an implicit question has been raised regarding how to do direction dependent (DD) ionospheric calibration: should one model the ionosphere on visibilities, or on gains?
Recently, we have discovered an unusual property related to fitting phase models on gains, which shows that sometimes simply adding more data leads to worse accuracy.
Recall that systematics, e.g. beam model errors, often manifest as very slowly changing phase offsets over the course of an observation.
Naturally, one might apply the prior that these systematics are fixed over time windows, to increase the signal of the ionospheric components.
We discover the non-intuitive fact that the posterior accuracy of the ionospheric components peaks at a certain window width and then actually gets worse with larger window sizes (as shown in Fig 1).
This unusual property is partly explained by the humped behaviour of the Fisher information.
