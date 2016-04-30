# From K-Means to Expectation Maximization

K-means is a simple clustering algorithm which matches each data point
to exactly one of K clusters in a way such that the sum of the squares
of the distances of each data point to the center of mass of its assigned
cluster is minimal. We review how this minimization can be performed
iteratively in a manner closely linked to Expectation Maximization for
Gaussian mixtures, then proceed with a deeper analysis of EM:

After introducing Gaussian mixtures with latent variables, we explain how EM
naturally appears for the estimation of the parameters, then apply it to
mixtures of Gaussian and Bernoulli variables. We also say a few words about the
Kullback-Leibler divergence to be able to show why EM works.

## Documentation

* The file `doc/km2em.pdf` and its source `doc/km2em.tm` contain a description
of the algorithm, a breif review of gaussian mixtures and some examples inside
live sessions of a [TeXmacs](http://www.texmacs.org) document.
* The file `doc/em.pdf` and its source `doc/em.tm` provide a more thorough
introduction to Expectation Maximization.

## Implementation (disclaimer)

We include a dirty Python implementation of the algorithms discussed, although
it's mostly oriented towards demo use inside TeXmacs: being simple exercises,
with innumerable available implementations online, the code is quite inelegant
and, being python, definitely slow. They certainly are un-pythonic, if anything
because I don't even know what qualifies as such. Don't look here for great coding
technique nor aesthetic pleasure.

## License

This software falls under the GNU general public license version 3 or later.
It comes without **any warranty whatsoever**.
For details see http://www.gnu.org/licenses/gpl-3.0.html.

