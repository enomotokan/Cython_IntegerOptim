from optimizer import *

""" 制約条件からグレーバー基底を生成し、保存したい場合 """
""" When you want to generate and save Graver basis from constraint conditions """

Solver = SCIOLIC()

# 実行可能解を最大何回探索するか How many times is a feasible solution searched
Solver.N_search_feasible = 10
# 近似度 approcimate accuracy
Solver.alpha = 0.9
# ログ出力するかどうか whether do or not log output
Solver.verbose = True

# 変数を定義する define variables
Solver.add_variable("x1", 3, 5)
Solver.add_variable("x2", 1, 4)
Solver.add_variable("x3", 2, 7)
Solver.add_variable("x4", 5, 7)
Solver.add_variable("x5", -3, 4)

# 制約条件を加える add constraint conditions
Solver.add_condition(">", 2)
Solver.define_coefficient(0, "x1", 5)
Solver.define_coefficient(0, "x2", -1)
Solver.define_coefficient(0, "x3", 3)
Solver.define_coefficient(0, "x4", 2)
Solver.define_coefficient(0, "x5", -4)


Solver.config_graver()

# 制約条件とグレーバー基底を保存する
Solver.save_conditions("prob1")

Solver.add_convexfunc("x2", 4)
Solver.add_convexfunc("x1", 3)
Solver.add_convexfunc("x3", 2)
Solver.add_convexfunc("x4", -2)
Solver.add_convexfunc("x5", -5)
print(Solver.search())