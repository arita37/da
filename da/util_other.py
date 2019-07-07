"""
Some special methods

"""


def optim_de(obj_fun, bounds, maxiter=1, name1="", solver1=None, isreset=1, popsize=15):
    """ Optimization and Save Data into file"""
    import copy

    if isreset == 2:
        print("Traditionnal Optim, no saving")
        res = sci.optimize.differential_evolution(obj_fun, bounds=bounds, maxiter=maxiter)
        xbest, fbest, solver, i = res.x, res.fun, "", maxiter
    else:  # iterative solver
        print("Iterative Solver ")
        if name1 != "":  # wtih file
            print("/batch/" + name1)
            solver2 = load_obj("/batch/" + name1)
            imin = int(name1[-3:]) + 1
            solver = sci.optimize._differentialevolution.DifferentialEvolutionSolver(
                obj_fun, bounds=bounds, popsize=popsize
            )
            solver.population = copy.deepcopy(solver2.population)
            solver.population_energies = copy.deepcopy(solver2.population_energies)
            del solver2

        elif solver1 is not None:  # Start from zero
            solver = copy.deepcopy(solver1)
            imin = 0
        else:
            solver = sci.optimize._differentialevolution.DifferentialEvolutionSolver(
                obj_fun, bounds=bounds, popsize=popsize
            )
            imin = 0

        name1 = "/batch/solver_" + name1
        fbest0 = 1500000.0
        for i in range(imin, imin + maxiter):
            xbest, fbest = next(solver)
            print(0, i, fbest, xbest)
            res = (copy.deepcopy(solver), i, xbest, fbest)
            try:
                util.save_obj(solver, name1 + util.date_now() + "_" + util.np_int_tostr(i))
                print((name1 + util.date_now() + "_" + util.np_int_tostr(i)))
            except:
                pass
            if np.mod(i + 1, 11) == 0:
                if np.abs(fbest - fbest0) < 0.001:
                    break
                fbest0 = fbest

    return fbest, xbest, solver


######################### OPTIM   ###################################################
def optim_is_pareto_efficient(Xmat_cost, epsilon=0.01, ret_boolean=1):
    """ Calculate Pareto Frontier of Multi-criteria Optimization program
    c1, c2  has to be minimized : -Sharpe, -Perf, +Drawdown
    :param Xmat_cost: An (n_points, k_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    pesp = 1.0 + epsilon  # Relax Pareto Constraints
    is_efficient = np.ones(Xmat_cost.shape[0], dtype=bool)
    for i, c in enumerate(Xmat_cost):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                Xmat_cost[is_efficient] <= c * pesp, axis=1
            )  # Remove dominated points
    if ret_boolean:
        return is_efficient
    else:
        return Xmat_cost[is_efficient]
    # return is_efficient


def pd_validation_struct():
    pass
    """
  https://github.com/jnmclarty/validada

  https://github.com/ResidentMario/checkpoints


  """


def pd_checkpoint():
    pass


"""
  Create Checkpoint on dataframe to save intermediate results
  https://github.com/ResidentMario/checkpoints
  To start, import checkpoints and enable it:

from checkpoints import checkpoints
checkpoints.enable()
This will augment your environment with pandas.Series.safe_map and pandas.DataFrame.safe_apply methods. Now suppose we create a Series of floats, except for one invalid entry smack in the middle:

import pandas as pd; import numpy as np
rand = pd.Series(np.random.random(100))
rand[50] = "____"
Suppose we want to remean this data. If we apply a naive map:

rand.map(lambda v: v - 0.5)

    TypeError: unsupported operand type(s) for -: 'str' and 'float'
Not only are the results up to that point lost, but we're also not actually told where the failure occurs! Using safe_map instead:

rand.safe_map(lambda v: v - 0.5)

    <ROOT>/checkpoint/checkpoints/checkpoints.py:96: UserWarning: Failure on index 50
    TypeError: unsupported operand type(s) for -: 'str' and 'float'


"""


"""
You can control how many decimal points of precision to display
In [11]:
pd.set_option('precision',2)

pd.set_option('float_format', '{:.2f}'.format)


Qtopian has a useful plugin called qgrid - https://github.com/quantopian/qgrid
Import it and install it.
In [19]:
import qgrid
qgrid.nbinstall()
Showing the data is straighforward.
In [22]:
qgrid.show_grid(SALES, remote_js=True)


SALES.groupby('name')['quantity'].sum().plot(kind="bar")


"""