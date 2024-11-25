import os
import time
import sys

# Get the directory containing readSMPS
readsmps_dir = "/Users/jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS"

# Add it to sys.path
if readsmps_dir not in sys.path:
    sys.path.append(readsmps_dir)

from readSMPS.decmps_2slp import RandVars
from readSMPS.decmps_2slp import decompose


def main():
    start = time.time()

    instance = "ssn"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    d = decompose(f"{instance}", input_dir)

    d.find_stage_idx()
    d.create_master()

    rand_vars = RandVars(d.name)

    print("TEST",len(rand_vars.observations))

    for i, obs in enumerate(rand_vars.observations, start=1):
        prob = rand_vars.probabilities[i - 1]
        d.create_sub(obs, prob, i)

    os.makedirs(output_dir, exist_ok=True)
    extensive_form_file = os.path.join(output_dir, "extensive_form.lp")
    # d.prob.extensive_form.write(extensive_form_file)

    d.prob.extensive_form.optimize()

    obj_value = d.prob.extensive_form.ObjVal
    print(f"Objective Value: {obj_value}")

    # Retrieve first-stage variable values
    for v in d.prob.extensive_form.getVars()[: d.prob.master_var_size]:
        print(f"{v.VarName} = {v.X}")


if __name__ == "__main__":
    main()
