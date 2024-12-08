# -*- coding: utf-8 -*-
"""
Created on May 4 - 2019

---Based on the 2-stage stochastic program structure
---Assumption: RHS is random
---decompose the 2slp problem into a master and subproblems
---save the distributoin of the random variables and return the
---random variables

@author: Siavash Tabrizian - stabrizian@smu.edu
"""

from readCOR import readcor
from readSTOC import readstoc
from readTIM import readtim

try:
    import gurobipy as gb
except ImportError as e:
    print("Gurobi is needed for building the problem structure!")
    raise


class prob:
    def __init__(self, name):
        # Import the Mean problem
        cor = readcor(name)
        cor.readfile()
        cor.get_mean()
        self.mean_model = cor.mean_model
        self.mean_vars = cor.mean_vars
        self.mean_const = cor.mean_const
        self.mean_sol = cor.mean_sol
        self.mean_status = cor.mean_status
        self.mean_objVal = cor.mean_objVal
        self.mean_var_size = cor.mean_var_num
        self.mean_const_size = cor.mean_const_num


class RandVars:
    def __init__(self, name):
        # Import stochastic information
        stoc = readstoc(name)
        stoc.readfile()
        self.rv = stoc.rv
        self.dist = stoc.dist
        self.cumul_dist = stoc.cumul_dist
        self.rvnum = stoc.rvnum


class TIME:
    def __init__(self, name):
        # Import time information
        tim = readtim(name)
        tim.readfile()
        self.stage_names = tim.stage
        self.stage_idx_col = list()
        self.stage_idx_row = list()
        self.stage_size = tim.stagenum


class decompose:
    def __init__(self, name, dirr):  # dirr = "./Input/"
        self.model_name = name
        self.name = dirr + name + "/" + name
        self.prob = prob(self.name)  # Prob information
        self.RV = RandVars(self.name)  # Random variabels
        self.tim = TIME(self.name)  # Time information(stages)

    # find the index of stages (rows and columns)
    def find_stage_idx(self):
        # initialize the index structs
        self.tim.stage_idx_col = list()
        self.tim.stage_idx_row = list()
        # find the column indexes (associated with variables)
        for i in range(0, self.tim.stage_size):
            count = 0
            for j in self.prob.mean_vars:
                if self.tim.stage_names[i][0] == j.varname:
                    self.tim.stage_idx_col.append(count)
                    break
                count += 1

        # find the row indexes (associated with constraints)
        for i in range(0, self.tim.stage_size):
            emp = [
                x
                for x in self.prob.mean_const
                if x.constrname == self.tim.stage_names[i][1]
            ]
            if emp == []:
                tmp = 0
            else:
                tmp = self.prob.mean_const.index(emp[0])
            # add the init and end index of the column of stage i
            self.tim.stage_idx_row.append(tmp)

    # replace the observations in the proper location of the problem
    def replaceObs(self, obs, constr):
        obscount = 0
        for c in constr:
            if c.getAttr("ConstrName") in self.RV.rv:
                if len(obs) < obscount - 1:
                    print("ERROR: Length of obs is less than rvs")
                    break
                else:
                    c.setAttr("RHS", obs[obscount])
                    self.prob.mean_model.update()
                    obscount += 1

        return constr

    # Create master constraints
    def create_master_constr(self):
        constr = self.prob.mean_const[: self.tim.stage_idx_row[1]]
        for c in constr:
            empt = gb.LinExpr()
            for v in self.prob.master_vars:
                empt += self.prob.mean_model.getCoeff(c, v) * v
            self.prob.extensive_form.addConstr(
                empt, c.getAttr("Sense"), c.getAttr("RHS"), c.getAttr("ConstrName")
            )
            self.prob.extensive_form.update()

    # Create subproblem constraints
    def create_sub_constr(self, obs, iteration):
        constr = self.prob.mean_const[self.tim.stage_idx_row[1] :]
        constr = self.replaceObs(obs, constr)

        for c in constr:
            empt = gb.LinExpr()
            for i, v in enumerate(self.prob.sub_vars):
                w = self.prob.sub_vars_fixed[i]
                empt += self.prob.mean_model.getCoeff(c, w) * v

            self.prob.extensive_form.addConstr(
                empt,
                c.getAttr("Sense"),
                c.getAttr("RHS"),
                name=f"{c.getAttr('ConstrName')}_{iteration}",
            )
            self.prob.extensive_form.update()

    def create_master(self, rep):
        # Start with an empty Gurobi model in each replication
        self.prob.extensive_form = gb.Model(f"extensive_form_{rep}")

        self.prob.master_vars = self.prob.mean_vars[: self.tim.stage_idx_col[1]]

        self.prob.master_var_size = len(self.prob.master_vars)

        # Create surrogate variables
        for v in self.prob.master_vars:
            self.prob.extensive_form.addVar(
                lb=v.getAttr("LB"),
                ub=v.getAttr("UB"),
                obj=v.getAttr("Obj"),
                vtype=v.getAttr("VType"),
                name=v.getAttr("VarName"),
            )

        self.prob.extensive_form.update()

        self.prob.master_vars = self.prob.extensive_form.getVars()

        self.create_master_constr()

    def create_sub(self, obs, prob, iteration):
        self.prob.sub_vars = self.prob.mean_vars
        self.prob.sub_vars_fixed = self.prob.sub_vars
        self.prob.sub_vars_s2 = self.prob.mean_vars[self.tim.stage_idx_col[1] :]

        self.prob.sub_var_s2_size = len(self.prob.sub_vars_s2)

        for v in self.prob.sub_vars_s2:
            self.prob.extensive_form.addVar(
                lb=v.getAttr("LB"),
                ub=v.getAttr("UB"),
                obj=prob * v.getAttr("Obj"),
                vtype=v.getAttr("VType"),
                name=f"{v.getAttr('VarName')}_{iteration}",  # give the vars a new name
            )

        self.prob.extensive_form.update()

        self.prob.sub_vars = self.prob.extensive_form.getVars()
        self.prob.sub_vars = (
            self.prob.sub_vars[: self.prob.master_var_size]
            + self.prob.sub_vars[-self.prob.sub_var_s2_size :]
        )
        self.prob.sub_vars_s2 = self.prob.sub_vars[-self.prob.sub_var_s2_size :]

        self.create_sub_constr(obs, iteration)

    # def create_LSsub(self, obs, incmb):
    #     self.prob.LSsub_vars = self.prob.mean_vars[self.tim.stage_idx_col[1] :]
    #     self.prob.LSsub_const = self.prob.mean_const[self.tim.stage_idx_col[1] :]

    #     for v in self.prob.LSsub_vars:
    #         self.prob.LSsub_model.addVar(
    #             lb=v.getAttr("LB"),
    #             ub=v.getAttr("UB"),
    #             obj=v.getAttr("Obj"),
    #             vtype=v.getAttr("VType"),
    #             name=v.getAttr("VarName"),
    #         )
    #     self.prob.LSsub_model.update()
    #     self.prob.LSsub_vars = self.prob.sub_model.getVars()

    #     # self.create_sub_constr(obs,incmb)
