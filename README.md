# K-Means and Expectation Maximization

K-means is a simple clustering algorithm which matches each data point
to exactly one of K clusters in a way such that the sum of the squares
of the distances of each data point to the center of mass of its assigned
cluster is minimal. We review how this minimization can be performed
iteratively in a manner closely linked to Expectation Maximization for
Gaussian mixtures. We also briefly discuss K-Means++.


## Documentation

Please read the contents of `doc` for a description of the algorithm and some
examples inside live sessions of a [TeXmacs](http://www.texmacs.org) document.

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

