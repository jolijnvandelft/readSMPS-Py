from decmps_2slp import decompose
import os

instance = "lands"
input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

def create_master(instance, input_dir):
    d = decompose(f"{instance}", input_dir)

    #Create master problem
    d.find_stage_idx()
    d.create_master()

    return d

d = create_master(instance, input_dir)

# Save the file in the newly created folder
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "Master_0.lp")
d.prob.master_model.write(output_file)