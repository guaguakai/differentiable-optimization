This is the implementation of the differentiable optimization layer.
This project features the ability to differentiate through a non-convex solver like scipy.minimize function.
We mainly use scipy.minimize as our non-convex solver with SLSQP algorithm.
Our wrapper takes a differentiable pytorch function (and constraint functions) as input.
It outputs a solution (potentially suboptimal) to minimize the given function and satisfy the constraints.
The output solution is differentiable, which can be chained with pytorch modeles or fed into a task-based loss function to train it end-to-end.

