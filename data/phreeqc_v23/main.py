import os
import time
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
import phreeqpy.iphreeqc.phreeqc_com as phreeqc_mod

def main():
    ### PROGRAM SETTINGS ###
    DEBUGGING = False
    MODEL = "model_v23.phr"
    if DEBUGGING:
        MODEL_SIZE = 10 # Maximum. Not guaranteed.
        THREADS = 1
    else:
        MODEL_SIZE = 1000
        
        # Leave 2 threads to OS.
        THREADS = os.process_cpu_count() - 2
        
        # Ensure the work is distributed homogenously.
        MODEL_SIZE -= MODEL_SIZE % THREADS
    
    ### MODEL INPUTS ###
    elements = ["ALKALINITY", "CARBONDIOXIDE", "HYDROGEN_SULFIDE", "QUARTZ"]

    # Generate folders if they do not exist.
    generate_folders()

    # Load default data and initialize error if debugging.
    if DEBUGGING:
        default_output = np.loadtxt("default_output.txt")
        error = 0

    # Run the model for the specified inputs.
    for i in range(0, MODEL_SIZE, THREADS):
        if DEBUGGING or THREADS == 1:
            # Report the generation number
            print(f"Running Generation {i + 1}...")
            
            # Run the model on at a time.
            start_time = time.time()
            output = work(i, elements, MODEL)
            print(f"Generation {i + 1} took {(time.time() - start_time):.2f} seconds to run.")

            # Generate plots for inspection.
            generate_plots(i, output[0], output[1:])

            # Generate a magnitude for both default and new output.
            # Try to normalize them to compansate different step count.
            observed = np.sum(output[1:]) / np.size(output[1:], axis=0)
            actual = np.sum(default_output) / np.size(default_output, axis=0)

            # Accumulate error
            error += (actual - observed) ** 2
        else:
            # Keep track of the processes.
            processes = []
            for j in range(THREADS):
                # Report the generation number
                print(f"Running Generation {i + j + 1}...")

                # Create a process.
                process = Process(target=work, args=(i + j, elements, MODEL))

                # Save the process then start it.
                processes.append(process)
                process.start()
            
            # Wait for the batch to conclude before moving on.
            for process in processes:
                process.join()
            
    # Report RMSE for sensitivity analysis if debugging.
    if DEBUGGING:
        rmse = (error / MODEL_SIZE) ** 0.5
        print(f"{elements} RMSE: {rmse}")

    return 0

def generate_folders():
    folders = ["input", "output", "plot"]

    for folder in folders:
        try:
            os.mkdir(f"./{folder}")
        except FileExistsError:
            pass
    return 0

def generate_inputs(elements):
    species = {
    "ALKALINITY": 4.90e-03,
    "SODIUM": 6.00e-04,
    "POTASSIUM": 1.50e-04,
    "MAGNESIUM": 4.40e-04,
    "CHLORINE": 3.00e-04,
    "IRON": 6.50e-06,
    "SULFUR": 3.70e-04,
    "CALCIUM": 4.10e-03,
    "FORMATE": 0.00e+00,
    "ACETATE": 7.00e-04,
    "METHANE": 70.0,
    "CARBONDIOXIDE": 2.9,
    "HYDROGEN": 9.2,
    "HYDROGEN_SULFIDE": 0.004,
    "QUARTZ": 0.080,
    "CALCITE": 0.012,
    "PYRITE": 0.002667,
    "BARITE": 0.002667,
    "ILLITE": 0.002667
    }

    rock_mass = 0.1 # kg

    for element in elements:
        if element in ["METHANE", "CARBONDIOXIDE", "HYDROGEN", "HYDROGEN_SULFIDE"]:
            species[element] *= np.random.uniform(0.5, 1.5)
            species["ACETATE"] = 1e-6
        elif element in ["QUARTZ", "CALCITE", "PYRITE", "BARITE", "ILLITE"]:
            species["QUARTZ"] = rock_mass * np.random.uniform(0.65, 0.95)
            remainder = rock_mass - species["QUARTZ"]
            species["CALCITE"] = remainder * np.random.uniform(0.3, 0.8)
            remainder = rock_mass - species["QUARTZ"] - species["CALCITE"]
            species["PYRITE"] = species["BARITE"] = species["ILLITE"] = remainder / 3
        elif element in ["ALKALINITY"]:
            species[element] *= np.random.uniform(0.5, 1.5)
        else:
            species[element] *= np.random.uniform(0.5, 1.5)

    return fix_rock_properties(species)

def fix_rock_properties(elements):
    species = {
    "QUARTZ": [60, 200, 2650],
    "CALCITE": [100, 100, 2710],
    "PYRITE": [120, 10, 5000],
    "BARITE": [234, 10, 4500],
    "ILLITE": [390, 1, 2700]
    }

    for specie in species:
        mass = elements[specie]
        molecular_weight = species[specie][0]
        diameter = species[specie][1] * 1e-6
        density = species[specie][2]
        elements[specie] = [(mass / molecular_weight * 1000), (6 * mass / density / diameter)]

    return elements

def format_inputs(elements):
    outputs = []
    for element in elements:
        if type(elements[element]) is list:
            outputs.append([(element + "_MOLES"), elements[element][0]])
            outputs.append([(element + "_AREA"), elements[element][1]])
        else:
            outputs.append([(element + "_VAL"), elements[element]])
    
    return outputs

def generate_legend(elements):
    np.savetxt(f"./output/0_legend.txt", elements, fmt="%s")
    return 0

def run_model(model, inputs):
    phreeqc = phreeqc_mod.IPhreeqc()
    phreeqc.load_database(r"phreeqc.dat")

    with open(model, 'r') as file:
        raw = file.read()
    
    for input in inputs:
        raw = raw.replace(input[0], str(input[1]))

    phreeqc.run_string(raw)

    selected_output = phreeqc.get_selected_output_array()

    return selected_output

def work(iter_num, elements, model):
    inputs = format_inputs(generate_inputs(elements))

    np.savetxt(f"./input/{iter_num + 1}_Input.txt", inputs, fmt="%s")

    output = run_model(model, inputs)

    np.savetxt(f"./output/{iter_num + 1}_Output.txt", output, fmt="%s")

    return output

def generate_plots(iter_num, headers, data):
    data = np.array(data, dtype=np.float64)
    for i in range(6):
        plt.plot(data[1:, 0], data[1:, i + 7] * 1000, label=f"{headers[i + 7]}")
    
    plt.xlabel("Time (days)")
    plt.ylabel("Species in Headspace (mmol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plot/{iter_num + 1}_Plot.jpg")
    plt.clf()

    return 0

if __name__ == '__main__':
    main()