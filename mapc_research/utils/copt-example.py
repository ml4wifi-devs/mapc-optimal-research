# Import PuLP and COPT PuLP plugin
from pulp import *
from copt_pulp import *

# Create LP problem
prob = LpProblem("lp_ex1", LpMaximize)

# Create variables
x = LpVariable("x", 0.1, 0.6)
y = LpVariable("y", 0.2, 1.5)
z = LpVariable("z", 0.3, 2.8)

# Create constraints
prob += 1.5*x + 1.2*y + 1.8*z <= 2.6, "c1"
prob += 0.8*x + 0.6*y + 0.9*z >= 1.2, "c2"

# Set objective
prob += 1.2*x + 1.8*y + 2.1*z, "obj"

# Solve problem
prob.solve(COPT_DLL())

# Print solution
print("\nOptimization status:", LpStatus[prob.status])
print("Objective value:", value(prob.objective))

print("Variable solution:")
for v in prob.variables():
  print(v.name, "=", v.varValue)
