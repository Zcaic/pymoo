import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.survival import Survival
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.core.survival import Survival

from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.lhs import LHS


class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F, cv = pop.get("F", "cv")
        assert F.shape[1] == 1, "FitnessSurvival can only used for single objective single!"
        S = np.lexsort([F[:, 0], cv])
        pop.set("rank", np.argsort(S))
        return pop[S[:n_survive]]


class JADE(Algorithm):
    """
    The original version of: Differential Evolution (JADE)

    Links:
        1. https://doi.org/10.1109/TEVC.2009.2014613

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial adaptive f, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial adaptive cr, default = 0.5
        + pt (float): [0.05, 0.2], The percent of top best agents (p in the paper), default = 0.1
        + ap (float): [0.05, 0.2], The Adaptation Parameter control value of f and cr (c in the paper), default=0.1

    References
    ~~~~~~~~~~
    [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional
    external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958.
    """

    def __init__(
        self,
        pop_size=50,
        miu_f=0.5,
        miu_cr=0.5,
        c=0.1,
        pt=0.1,
        sampling=LHS(),
        repair=NoRepair(),
        output=None,
        display=None,
        callback=None,
        archive=None,
        return_least_infeasible=False,
        save_history=False,
        verbose=False,
        seed=None,
        evaluator=None,
        **kwargs,
    ):
        self.pop_size = pop_size
        self.repair = repair
        self.survial = FitnessSurvival()
        self.initialization = Initialization(sampling, self.repair)

        self.pt = pt
        self.c = c
        self.dyn_miu_cr = miu_cr
        self.dyn_miu_f = miu_f
        self.dyn_pop_archive = Population()

        super().__init__(
            termination=None,
            output=output,
            display=display,
            callback=callback,
            archive=archive,
            return_least_infeasible=return_least_infeasible,
            save_history=save_history,
            verbose=verbose,
            seed=seed,
            evaluator=evaluator,
            **kwargs,
        )

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survial.do(self.problem, infills)

    def gen_cr(self):
        cr = self.random_state.normal(self.dyn_miu_cr, 0.1, self.pop_size)
        cr = np.clip(cr, 0, 1)
        return cr

    def gen_f(self):
        # while 1:
        #     f = cauchy.rvs(self.dyn_miu_f, 0.1, random_state=self.random_state)
        #     if f < 0.0:
        #         continue
        #     elif f > 1.0:
        #         f = 1
        #     break
        # return f

        f = np.clip(self.dyn_miu_f + np.sqrt(0.1) * np.random.standard_cauchy(self.pop_size), 0.000001, 1)
        return f

    def _infill(self):
        ndim = self.problem.n_var

        X = self.pop.get("X")
        off = []

        self.list_f = []
        self.list_cr = []
        self.temp_cr = self.gen_cr()
        self.temp_f = self.gen_f()

        for idx in range(self.pop_size):
            cr = self.temp_cr[idx]
            f = self.temp_f[idx]

            top_range = int(self.pt * self.pop_size)
            random_best = self.random_state.integers(0, top_range)
            best_p_x = X[random_best]

            r1_idx = self.random_state.choice(list(set(range(0, self.pop_size)) - {idx}))
            new_pop = Population.merge(self.pop, self.dyn_pop_archive)
            r2_idx = self.random_state.choice(list(set(range(0, len(new_pop))) - {idx, r1_idx}))

            x_r1 = X[r1_idx]
            x_r2 = new_pop[r2_idx].get("X")

            current_x = X[idx]

            # mutation
            x_new = current_x + f * (best_p_x - current_x) + f * (x_r1 - x_r2)

            # cross
            pos_new = np.where(self.random_state.random(ndim) < cr, x_new, current_x)
            j_rand = self.random_state.integers(0, ndim)
            pos_new[j_rand] = x_new[j_rand]

            off.append(pos_new)

        off = np.array(off)
        if self.problem.has_bounds():
            # off = set_to_bounds_if_outside(off, *self.problem.bounds())
            x_new = repair_random_init(off, X, *self.problem.bounds(), random_state=self.random_state)

        off = Population.new(X=off)
        off = self.repair.do(self.problem, off)
        return off

    def _advance(self, infills=None, **kwargs):
        off = infills
        pop = self.pop
        has_improved = np.full((self.pop_size, 1), False)

        pop_F, pop_CV, pop_feas = pop.get("F", "CV", "FEAS")
        off_F, off_CV, off_feas = off.get("F", "CV", "FEAS")

        if self.problem.has_constraints() > 0:

            # 1) Both infeasible and constraints have been improved
            has_improved[(~pop_feas & ~off_feas) & (off_CV < pop_CV)] = True

            # 2) A solution became feasible
            has_improved[~pop_feas & off_feas] = True

            # 3) Both feasible but objective space value has improved
            has_improved[(pop_feas & off_feas) & (off_F < pop_F)] = True

        else:
            has_improved[off_F < pop_F] = True

        # never allow duplicates to become part of the population when replacement is used
        _, _, is_duplicate = DefaultDuplicateElimination(epsilon=0.0).do(off, pop, return_indices=True)
        has_improved[is_duplicate] = False
        has_improved = has_improved[:, 0]

        self.dyn_pop_archive = Population.merge(self.dyn_pop_archive, pop[has_improved].copy())
        self.list_cr = self.temp_cr[has_improved].copy()
        self.list_f = self.temp_f[has_improved].copy()
        pop[has_improved] = off[has_improved].copy()

        len_dyn_pop_archive = len(self.dyn_pop_archive)
        temp = len_dyn_pop_archive - self.pop_size
        if temp > 0:
            mask = np.full(len_dyn_pop_archive, True)
            idx_to_remove = self.random_state.choice(range(0, len_dyn_pop_archive), temp, replace=False)
            mask[idx_to_remove] = False
            self.dyn_pop_archive = self.dyn_pop_archive[mask]

        if len(self.list_cr) == 0:
            self.dyn_miu_cr = (1 - self.c) * self.dyn_miu_cr + self.c * 0.5
        else:
            self.dyn_miu_cr = (1 - self.c) * self.dyn_miu_cr + self.c * np.mean(self.list_cr)

        if len(self.list_f) == 0:
            self.dyn_miu_f = (1 - self.c) * self.dyn_miu_f + self.c * 0.5
        else:
            self.dyn_miu_f = (1 - self.c) * self.dyn_miu_f + self.c * self.lehmer_mean(self.list_f)

        self.survial.do(self.problem, pop)

    def lehmer_mean(self, list_objects):
        temp = np.sum(list_objects)
        return 0 if temp == 0 else np.sum(list_objects**2) / temp


if __name__ == "__main__":
    from pymoo.problems.single.rosenbrock import Rosenbrock
    from pymoo.algorithms.soo.nonconvex.nrbo import NRBO
    from pymoo.algorithms.soo.nonconvex.de import DE

    prob = Rosenbrock(n_var=10)
    algo = JADE(pop_size=30)
    # algo=DE(pop_size=30)
    # algo=NRBO(pop_size=30,max_iteration=100)

    algo.setup(problem=prob, termination=("n_gen", 1000), seed=2)

    while algo.has_next():
        pop = algo.ask()
        algo.evaluator.eval(problem=algo.problem, pop=pop)
        algo.tell(infills=pop)

    result = algo.result()
    print(result.F)
    print(result.X)

    # from mealpy import FloatVar, DE

    # def objective_function(solution):
    #     solution = solution[None, :]
    #     return prob.evaluate(solution)[0, 0]

    # problem_dict = {
    #     "bounds": FloatVar(lb=prob.xl, ub=prob.xu, name="delta"),
    #     "minmax": "min",
    #     "obj_func": objective_function,
    #     "log_to": None,
    # }

    # model = DE.JADE(epoch=1000, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1)
    # g_best = model.solve(problem_dict, seed=1)
    # print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    # print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
