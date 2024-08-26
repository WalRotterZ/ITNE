import gurobipy as gp
import pickle

model = gp.Model(name='ITNE')

x01, x02 = model.addVar(lb=-1, ub=1, name='x01'), model.addVar(lb=1, ub=2, name='x02')
_x01, _x02 = model.addVar(lb=-1, ub=1, name='_x01'), model.addVar(lb=1, ub=2, name='_x02')
y11, y12, y2 = model.addVar(lb=-gp.GRB.INFINITY, name='y11'), model.addVar(lb=-gp.GRB.INFINITY, name='y12'), model.addVar(lb=-gp.GRB.INFINITY, name='y2')
_y11, _y12, _y2 = model.addVar(lb=-gp.GRB.INFINITY, name='_y11'), model.addVar(lb=-gp.GRB.INFINITY, name='_y12'), model.addVar(lb=-gp.GRB.INFINITY, name='_y2')
x11, x12, x2 = model.addVar(name='x11'), model.addVar(name='x12'), model.addVar(name='x2')
_x11, _x12, _x2 = model.addVar(name='_x11'), model.addVar(name='_x12'), model.addVar(name='_x2')
delta_y11, delta_y12, delta_y2 = model.addVar(lb=-gp.GRB.INFINITY, name='dy11'), model.addVar(lb=-gp.GRB.INFINITY, name='dy12'), model.addVar(lb=-gp.GRB.INFINITY, name='dy2')
delta_x11, delta_x12, delta_x2 = model.addVar(lb=-gp.GRB.INFINITY, name='dx11'), model.addVar(lb=-gp.GRB.INFINITY, name='dx12'), model.addVar(lb=-gp.GRB.INFINITY, name='dx2')

model.addConstr(x01 - _x01 <= 0.1)
model.addConstr(x01 - _x01 >= -0.1)
model.addConstr(x02 - _x02 <= 0.1)
model.addConstr(x02 - _x02 >= -0.1)


m = int(input("Select Model(1,2,3): "))

if m == 1:
    # Original Neural Network
    model.addConstr(y11 == 1.0 * x01 + 0.5 * x02)
    model.addConstr(y12 == -0.5 * x01 + 1.0 * x02)
    model.addConstr(_y11 == 1.0 * _x01 + 0.5 * _x02)
    model.addConstr(_y12 == -0.5 * _x01 + 1.0 * _x02)
elif m == 2:
    # Ly11 > 0
    model.addConstr(y11 == 1.0 * x01 + 1.0 * x02)
    model.addConstr(y12 == 2.0 * x01 + 1.0 * x02)
    model.addConstr(_y11 == 1.0 * _x01 + 1.0 * _x02)
    model.addConstr(_y12 == 2.0 * _x01 + 1.0 * _x02)
elif m == 3:
    # Uy12 < 0
    model.addConstr(y11 == 1.0 * x01 + 1.0 * x02)
    model.addConstr(y12 == 2.0 * x01 - 2.0 * x02)
    model.addConstr(_y11 == 1.0 * _x01 + 1.0 * _x02)
    model.addConstr(_y12 == 2.0 * _x01 - 2.0 * _x02)


model.addConstr(y2 == -1.0 * x12 + 1.0 * x11)
model.addConstr(_y2 == -1.0 * _x12 + 1.0 * _x11)

model.addConstr(delta_y11 == y11 - _y11)
model.addConstr(delta_y12 == y12 - _y12)
model.addConstr(delta_x11 == x11 - _x11)
model.addConstr(delta_x12 == x12 - _x12)
model.addConstr(delta_y2 == y2 - _y2)
model.addConstr(delta_x2 == x2 - _x2)

model.setObjective(y11, gp.GRB.MAXIMIZE)
model.optimize()
u_y11 = y11.X
model.setObjective(y11, gp.GRB.MINIMIZE)
model.optimize()
l_y11 = y11.X
model.setObjective(y12, gp.GRB.MAXIMIZE)
model.optimize()
u_y12 = y12.X
model.setObjective(y12, gp.GRB.MINIMIZE)
model.optimize()
l_y12 = y12.X

model.addConstr(x11 >= 0)
model.addConstr(x11 >= y11)
model.addConstr((u_y11 - l_y11) * x11 <= u_y11 * (y11 - l_y11))
model.addConstr(x12 >= 0)
model.addConstr(x12 >= y12)
model.addConstr((u_y12 - l_y12) * x12 <= u_y12 * (y12 - l_y12))

model.setObjective(delta_y11, gp.GRB.MAXIMIZE)
model.optimize()
u_delta_y11 = delta_y11.X
model.setObjective(delta_y11, gp.GRB.MINIMIZE)
model.optimize()
l_delta_y11 = delta_y11.X
model.setObjective(delta_y12, gp.GRB.MAXIMIZE)
model.optimize()
u_delta_y12 = delta_y12.X
model.setObjective(delta_y12, gp.GRB.MINIMIZE)
model.optimize()
l_delta_y12 = delta_y12.X

if m == 1:
    model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 <= u_delta_y11 * (delta_y11 - l_delta_y11))
    model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 >= l_delta_y11 * (u_delta_y11 - delta_y11))
    model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 <= u_delta_y12 * (delta_y12 - l_delta_y12))
    model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 >= l_delta_y12 * (u_delta_y12 - delta_y12))
elif m == 2:
    n = int(input("Select Constraint Method(1:Original 2:Reduction): "))
    if n == 1:
        model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 <= u_delta_y11 * (delta_y11 - l_delta_y11))
        model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 >= l_delta_y11 * (u_delta_y11 - delta_y11))
        model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 <= u_delta_y12 * (delta_y12 - l_delta_y12))
        model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 >= l_delta_y12 * (u_delta_y12 - delta_y12))
    elif n == 2:
        model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 <= u_delta_y11 * (delta_y11 - l_delta_y11))
        model.addConstr(delta_x11 >= delta_y11)
        model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 <= u_delta_y12 * (delta_y12 - l_delta_y12))
        model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 >= l_delta_y12 * (u_delta_y12 - delta_y12))
elif m == 3:
    n = int(input("Select Constraint Method(1:Original 2:Reduction): "))
    if n == 1:
        model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 <= u_delta_y11 * (delta_y11 - l_delta_y11))
        model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 >= l_delta_y11 * (u_delta_y11 - delta_y11))
        model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 <= u_delta_y12 * (delta_y12 - l_delta_y12))
        model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 >= l_delta_y12 * (u_delta_y12 - delta_y12))
    elif n == 2:
        model.addConstr((u_delta_y11 - l_delta_y11) * delta_x11 <= u_delta_y11 * (delta_y11 - l_delta_y11))
        model.addConstr(delta_x11 >= delta_y11)
        model.addConstr((u_delta_y12 - l_delta_y12) * delta_x12 <= u_delta_y12 * (delta_y12 - l_delta_y12))
        model.addConstr(delta_x12 >= 0)


model.setObjective(y2, gp.GRB.MAXIMIZE)
model.optimize()
u_y2 = y2.X
model.setObjective(y2, gp.GRB.MINIMIZE)
model.optimize()
l_y2 = y2.X

model.addConstr(x2 >= 0)
model.addConstr(x2 >= y2)
model.addConstr((u_y2 - l_y2) * x2 <= u_y2 * (y2 - l_y2))


model.setObjective(delta_y2, gp.GRB.MAXIMIZE)
model.optimize()
u_delta_y2 = delta_y2.X
model.setObjective(delta_y2, gp.GRB.MINIMIZE)
model.optimize()
l_delta_y2 = delta_y2.X

model.addConstr((u_delta_y2 - l_delta_y2) * delta_x2 <= u_delta_y2 * (delta_y2 - l_delta_y2))
model.addConstr((u_delta_y2 - l_delta_y2) * delta_x2 >= l_delta_y2 * (u_delta_y2 - delta_y2))

u_obj = model.addVar(lb = -gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS, name = "U_OBJ")
l_obj = model.addVar(lb = -gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS, name = "L_OBJ")

model.addConstr(u_obj == delta_x2)
model.setObjective(u_obj, sense=gp.GRB.MAXIMIZE)
model.optimize()
if model.Status == gp.GRB.OPTIMAL:
    obj_opt = u_obj.X
    print(f"Upper Bound = {obj_opt}")

model.addConstr(l_obj == delta_x2)
model.setObjective(l_obj, sense=gp.GRB.MINIMIZE)
model.optimize()
if model.Status == gp.GRB.OPTIMAL:
    obj_opt = l_obj.X
    print(f"Lower Bound = {obj_opt}")