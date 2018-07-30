# CGP optimization methods

Experiments with different optimization methods for Cartesian Genetic Programming.

Uses [tensor-cgp](https://github.com/Jarino/tensor-cgp) and [gpbenchmarks](https://github.com/Jarino/gpbenchmarks).

Version of tensor-cgp used in notebook experiment is always included in subfolder.

## Tabu search #1

For single mutation, very simple short-term memory was created. Move was represented as two lists:

- first contained indices of changed genes,
- second contained new values for corresponding indices from first list.

Furthermore, each performed move was saved as tabu, whether it was better or worse. Results for memory of size 1000 moves:

| statistic | basic single mutation | tabu search single mutation |
|-----------|-----------------------|-----------------------------|
| mean      | 0.0398                | 0.0371                      |
| median    | 0.0056                | 0.0059                      |
| hit count (strict) | 21           | 4                           |
| hit count (approx) | 4            | 3                           |

When smaller memory was used (100 moves), results were worse (mean 0.11, median 0.0066).

Observation: When using 1000 moves as a memory size, in each run about 100 moves were stored (rough estimate). If these moves would be applied, it almost never resulted in worse fitness (maximum was 3 worsening applications during one run, usually there were none worsening applications). 

## Tabu search #2

In this case, move was represented as a single list containing new values (without gene index specified). Move was therefore rendered tabu, if it were to change genes at whatever position to values stored in memory.

| statistic | basic single mutation | tabu search single mutation |
|-----------|-----------------------|-----------------------------|
| mean      | 0.0398                | 0.0268                      |
| median    | 0.0056                | 0.0054                      |
| hit count (strict) | 21           | 4                           |
| hit count (approx) | 21           | 0                           |

Number of memory hits is currently not known.

## Tabu search #3

It is the same as tabu search #2, only move is evaluated also when its tabu and if the fitness is better or the same, the move is applied (kind of simple aspiration criteria).

Since the number of memory hits was approx. around 1/7 of all moves, the impact of more function evaluations should not be that bad.

But the performance was slightly worse than TS#2 on Nguyen-4 problem:

Best achieved: 1.3789658401812608e-32
Mean: 0.07354103095024606
Median: 0.02232148956290639

In comparison, the TS#2 on  Nguyen-4:
Best achieved: 1.1095763892100962e-32
Mean: 0.05563121474806537	
Median: 0.01769006375046512


## Tabu search #4

Even less strict tabu rule in TS#2 did not lead to significant amount of prohibited good moves (and furthermore, even their allowance did not help). On the other side, almost every memory hit prevents worse solution (the ration of hits and worse solutions is 1.13).

This suggests, that even less strict tabu rule could be employed. Since we already do not care about the indices of changes, we could relax changes only to active genes of given individual.
