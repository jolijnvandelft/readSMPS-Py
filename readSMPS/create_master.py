from decmps_2slp import decompose
import os

instance = "20"
output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

d = decompose(f"{instance}", "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/")

os.makedirs(output_dir, exist_ok=True)

#Create master problem
d.find_stage_idx()
d.create_master()
d.prob.master_model.write(f"Master_{instance}_0.lp")

# Save the file in the newly created folder
output_file = os.path.join(output_dir, f"Master_{instance}_0.lp")
d.prob.master_model.write(output_file)
print(f"File saved to: {output_file}")