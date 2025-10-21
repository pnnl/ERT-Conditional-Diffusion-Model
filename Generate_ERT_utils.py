import os
import numpy as np
import subprocess
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt

class ParameterLimits:
    def __init__(self):
        self.plims = np.ones((29, 2))
        self._initialize_limits()
        self.logparms = [1, 7, 13, 18, 21, 24, 25, 26, 27, 28]

    def _initialize_limits(self):
        # Material 1
        self.plims[0:6] = [
            [.05, .35],    # porosity
            [2e-13, 1e-7], # horizontal permeability
            [0.2, 1],      # vertical/horizontal permeability ratio
            [1.1, 2.1],    # archies cementation exponent
            [1.6, 2.6],    # archies saturation exponent
            [0.8, 1.2]     # archies tortuosity constant
        ]

        # Material 2
        self.plims[6:12] = [
            [.2, .5],      # porosity
            [1e-12, 1e-8], # horizontal permeability
            [0.2, 1],      # vertical/horizontal permeability ratio
            [1.1, 2.1],    # archies cementation exponent
            [1.6, 2.6],    # archies saturation exponent
            [0.8, 1.2]     # archies tortuosity constant
        ]

        # Material 3
        self.plims[12:18] = [
            [.05, .35],    # porosity
            [2e-13, 1e-7], # horizontal permeability
            [0.2, 1],      # vertical/horizontal permeability ratio
            [1.1, 2.1],    # archies cementation exponent
            [1.6, 2.6],    # archies saturation exponent
            [0.8, 1.2]     # archies tortuosity constant
        ]

        # Van Genuchten parameters
        self.plims[18:24] = [
            [2e-5, 9e-3],  # Hanford Formation VG-Alpha
            [.2, .65],     # Hanford Formation VG-M
            [.0055, .24],  # Hanford Formation Liquid residual saturation
            [1e-5, 8e-3],  # Ringold Formation VG-Alpha
            [.16, .8],     # Ringold Formation VG-M
            [.02, .2]      # Ringold Formation Liquid residual saturation
        ]

        # Surface electrical conductivities
        self.plims[24:27] = [[1e-5, 1e-2]] * 3

        # Water conductivity
        self.plims[27:29] = [[0.005, 0.1]] * 2  # pore and flush water conductivity

class ParameterNames:
    def __init__(self):
        self.names = []
        self._initialize_names()

    def _initialize_names(self):
        self.names = [
            "Hanford Fm porosity",
            "Hanford Fm horizontal permeability [$m^2$]",
            "Hanford Fm vertical/horiz. perm. ratio",
            "Hanford Fm Archie's law cementation exponent",
            "Hanford Fm Archie's law saturation exponent",
            "Hanford Fm Archie's law tortuosity constant",
            "Ringold Fm unit porosity",
            "Ringold Fm horizontal permeability [$m^2$]",
            "Ringold Fm vertical/horiz. perm. ratio",
            "Ringold Fm Archie's law cementation exponent",
            "Ringold Fm Archie's law saturation exponent",
            "Ringold Fm Archie's law tortuosity constant",
            "Pit porosity",
            "Pit horizontal permeability [$m^2$]",
            "Pit vertical/horiz. perm. ratio",
            "Pit Archie's law cementation exponent",
            "Pit Archie's law saturation exponent",
            "Pit Archie's law tortuosity constant",
            "Hanford Fm and Pit  VG-Alpha [$1/m$]",
            "Hanford Fm and Pit  VG-M",
            "Hanford Fm and Pit residual saturation",
            "Ringold Fm VG-Alpha [$1/m$]",
            "Ringold Fm VG-M",
            "Ringold Fm residual saturation",
            "Hanford Fm surface electrical conductivity [$S/m$]",
            "Ringold Fm surface electrical conductivity [$S/m$]",
            "Pit surface electrical conductivity [$S/m$]",
            "Native pore water conductivity [$S/m$]",
            "Flush water conductivity [$S/m$]"]
        

class ParameterScaler:
    def __init__(self, param_limits):
        self.param_limits = param_limits

    def scale(self, pm):
        scpm = np.zeros(len(pm))
        for i in range(len(pm)):
            p = self.param_limits.plims[i]
            scpm[i] = (1/(p[1]-p[0]))*(pm[i]-p[0])
        
        for i in self.param_limits.logparms:
            p = self.param_limits.plims[i]
            pmin = np.log10(p[0])
            pmax = np.log10(p[1])
            px = np.log10(pm[i])
            scpm[i] = (1/(pmax-pmin))*(px-pmin)
        return scpm

    def descale(self, pm):
        scpm = np.zeros(len(pm))
        for i in range(len(pm)):
            p = self.param_limits.plims[i]
            scpm[i] = (p[1]-p[0])*(pm[i]) + p[0]
            
        for i in self.param_limits.logparms:
            p = self.param_limits.plims[i]
            pmin = np.log10(p[0])
            pmax = np.log10(p[1])
            scpm[i] = 10**((pmax-pmin)*(pm[i])+ pmin)

        self._check_limits(scpm)
        return scpm

    def _check_limits(self, scpm):
        for i in range(len(scpm)):
            if scpm[i] < self.param_limits.plims[i,0]:
                scpm[i] = self.param_limits.plims[i,0]
                print(f'Warning: Parameter {i} is set to the minimum allowable of {scpm[i]}')
            if scpm[i] > self.param_limits.plims[i,1]:
                scpm[i] = self.param_limits.plims[i,1]
                print(f'Warning: Parameter {i} is set to the maximum allowable of {scpm[i]}')

class ParameterSampler:
    def __init__(self, param_limits):
        self.param_limits = param_limits

    def get_mean_parameters(self):
        return np.array([np.average(self.param_limits.plims[i]) for i in range(29)])

    def sample(self):
        pm = np.zeros(29)
        for i, p in enumerate(self.param_limits.plims):
            pm[i] = np.random.uniform(p[0], p[1])

        for i in self.param_limits.logparms:
            p = self.param_limits.plims[i]
            pm[i] = 10**(np.random.uniform(np.log10(p[0]), np.log10(p[1])))

        self._check_limits(pm)
        return pm

    def build_mod_in(self, pmeans, psdevs, nmods):
        """Build model input parameters with given means and standard deviations.
        
        Args:
            pmeans: Array of mean values for each parameter
            psdevs: Either a single float (same std dev for all parameters) or array of std devs
            nmods: Number of models to generate
        """
        mod_in = np.zeros((nmods, 29))
        mod_in[0] = pmeans
        
        # Convert psdevs to array if it's a single value
        if isinstance(psdevs, (float, int)):
            psdevs = np.ones(29) * psdevs
        
        for s in range(1, nmods):
            failed = np.ones(29, dtype='bool')
            while any(failed):
                inds = np.where(failed)[0]
                for i in inds:
                    mod_in[s,i] = np.random.normal(pmeans[i], psdevs[i], 1)[0]
                    
                    if i in self.param_limits.logparms:
                        sd = abs(np.log10(pmeans[i])*psdevs[i]/pmeans[i])
                        lval = np.random.normal(np.log10(pmeans[i]), sd, 1)[0]
                        mod_in[s,i] = 10**lval
                
                for i in inds:
                    plims = self.param_limits.plims[i]
                    if (mod_in[s,i] > plims[0]) and (mod_in[s,i] < plims[1]):
                        failed[i] = False
        
        return mod_in

    def _check_limits(self, pm):
        for i in range(29):
            if pm[i] < self.param_limits.plims[i,0]:
                pm[i] = self.param_limits.plims[i,0]
                print(f'Warning: Parameter {i} is set to the minimum allowable of {pm[i]}')
            if pm[i] > self.param_limits.plims[i,1]:
                pm[i] = self.param_limits.plims[i,1]
                print(f'Warning: Parameter {i} is set to the maximum allowable of {pm[i]}')

class ERTDataHandler:
    @staticmethod
    def gather_data(prefix):
        """Gather ERT data from .srv files with given prefix."""
        # Get all relevant files
        fnames = [f for f in os.listdir('./') 
                 if f.startswith(prefix) and f.endswith('.srv')]
        
        # Extract times and sort
        times = [float(f.split('-')[2].split('d')[0]) for f in fnames]
        order = np.argsort(times)
        
        # Read data from files
        data = []
        for i in order:
            fil = fnames[i]
            di = np.genfromtxt(fil, skip_header=259, usecols=5)
            data.append(di)
        data = np.array(data)
        
        # Flatten the data
        return data.flatten()

class PflotranSimulator:
    def __init__(self, template_file, output_prefix):
        self.template_file = template_file
        self.output_prefix = output_prefix
        self.template_lines = self._read_template()
        
    def _read_template(self):
        """Read the PFLOTRAN input file template."""
        with open(self.template_file, 'r') as f:
            return f.readlines()
    
    def _create_input_file(self, parameters, model_index):
        """Create a PFLOTRAN input file for a specific parameter set."""
        lines = self.template_lines.copy()
        
        # Convert parameters to strings for writing
        str_params = [f"{p:12.4e} \n" for p in parameters]
        
        # Material 1 parameters
        lines[79:83] = [
            f"  ARCHIE_CEMENTATION_EXPONENT {str_params[3]}",
            f"  ARCHIE_SATURATION_EXPONENT  {str_params[4]}",
            f"  ARCHIE_TORTUOSITY_CONSTANT  {str_params[5]}",
            f"  POROSITY {str_params[0]}"
        ]
        lines[87:90] = [
            f"    PERM_X {str_params[1]}",
            f"    PERM_Y {str_params[1]}",
            f"    PERM_Z {parameters[1]*parameters[2]} \n"
        ]
        
        # Material 2 parameters
        lines[97:101] = [
            f"  ARCHIE_CEMENTATION_EXPONENT {str_params[9]}",
            f"  ARCHIE_SATURATION_EXPONENT  {str_params[10]}",
            f"  ARCHIE_TORTUOSITY_CONSTANT  {str_params[11]}",
            f"  POROSITY {str_params[6]}"
        ]
        lines[105:108] = [
            f"    PERM_X {str_params[7]}",
            f"    PERM_Y {str_params[7]}",
            f"    PERM_Z {parameters[7]*parameters[8]} \n"
        ]
        
        # Material 3 parameters
        lines[115:119] = [
            f"  ARCHIE_CEMENTATION_EXPONENT {str_params[15]}",
            f"  ARCHIE_SATURATION_EXPONENT  {str_params[16]}",
            f"  ARCHIE_TORTUOSITY_CONSTANT  {str_params[17]}",
            f"  POROSITY {str_params[12]}"
        ]
        lines[123:126] = [
            f"    PERM_X {str_params[13]}",
            f"    PERM_Y {str_params[13]}",
            f"    PERM_Z {parameters[13]*parameters[14]} \n"
        ]
        
        # Van Genuchten parameters
        self._set_van_genuchten_params(lines, str_params)
        
        # Conductivity parameters
        self._set_conductivity_params(lines, str_params)
        
        # Write the input file
        output_file = f"{self.output_prefix}_{model_index:04d}.in"
        with open(output_file, 'w') as f:
            f.writelines(lines)
        
        return output_file
    
    def _set_van_genuchten_params(self, lines, str_params):
        """Set Van Genuchten parameters in the input file."""
        # Hanford Formation
        lines[133:136] = [
            f"    ALPHA {str_params[18]}",
            f"    M {str_params[19]}",
            f"    LIQUID_RESIDUAL_SATURATION {str_params[20]}"
        ]
        lines[140:142] = [
            f"    M {str_params[19]}",
            f"    LIQUID_RESIDUAL_SATURATION {str_params[20]}"
        ]
        
        # Ringold Formation
        lines[147:150] = [
            f"    ALPHA {str_params[21]}",
            f"    M {str_params[22]}",
            f"    LIQUID_RESIDUAL_SATURATION {str_params[23]}"
        ]
        lines[154:156] = [
            f"    M {str_params[22]}",
            f"    LIQUID_RESIDUAL_SATURATION {str_params[23]}"
        ]
    
    def _set_conductivity_params(self, lines, str_params):
        """Set conductivity parameters in the input file."""
        lines[84] = f"  SURFACE_ELECTRICAL_CONDUCTIVITY {str_params[24]}"
        lines[102] = f"  SURFACE_ELECTRICAL_CONDUCTIVITY {str_params[25]}"
        lines[120] = f"  SURFACE_ELECTRICAL_CONDUCTIVITY {str_params[26]}"
        lines[22] = f"        WATER_CONDUCTIVITY {str_params[27]}"

class ForwardModelRunner:
    def __init__(self, pflotran_simulator, parameter_sampler, ert_handler, pflotran_path=None):
        self.simulator = pflotran_simulator
        self.parameter_sampler = parameter_sampler
        self.ert_handler = ert_handler
        if pflotran_path is None:
            self.pflotran_path = "pflotran/src/pflotran/pflotran"
        else:
            self.pflotran_path = pflotran_path

    def run_simulations(self, pmean, psdev, n_models):
        """Run forward simulations with given parameters."""
        failed = np.ones(n_models, dtype='bool')
        parameters = np.zeros((n_models, 29))
        data = np.zeros((n_models, 37544))
        
        print(f"\nStarting simulation batch of {n_models} models...")
        print("=" * 50)
        
        attempt = 1
        while any(failed):
            failed_indices = np.where(failed)[0]
            if attempt > 1:
                print(f"\nRetrying failed simulations. Attempt {attempt}")
                print(f"Failed indices: {failed_indices}")
            
            parameters = self.parameter_sampler.build_mod_in(
                pmean, psdev, n_models)
            
            self._run_parallel_simulations(failed_indices, parameters)
            
            # Collect results
            for i in failed_indices:
                prefix = f"{self.simulator.output_prefix}_{i:04d}"
                try:
                    data[i] = self.ert_handler.gather_data(prefix)
                    failed[i] = False
                    print(f"Successfully completed simulation {i+1}/{n_models}")
                except:
                    print(f"WARNING: Simulation {i+1}/{n_models} failed and will be retried")
            
            attempt += 1
            
            # Print current progress
            success_count = n_models - np.sum(failed)
            print(f"\nProgress: {success_count}/{n_models} simulations completed successfully")
            print("=" * 50)
        
        print("\nAll simulations completed successfully!")
        return parameters, data

    def _run_parallel_simulations(self, indices, parameters):
        """Run PFLOTRAN simulations in parallel."""
        batch_size = 6
        total_batches = (len(indices) + batch_size - 1) // batch_size  # ceiling division
        
        print(f"\nProcessing {len(indices)} simulations in {total_batches} batches")
        
        for batch_num, i in enumerate(range(0, len(indices), batch_size), 1):
            batch_indices = indices[i:i+batch_size]
            print(f"\nStarting batch {batch_num}/{total_batches}")
            print(f"Batch indices: {batch_indices}")
            self._run_batch(batch_indices, parameters)

    def _run_batch(self, batch_indices, parameters):
        """Run a batch of simulations."""
        script_content = ['#!/bin/bash']
        pflotran_cmd = f"mpirun -np 6 {self.pflotran_path} -pflotranin"
        
        for idx in batch_indices:
            input_file = self.simulator._create_input_file(parameters[idx], idx)
            script_content.append(f"{pflotran_cmd} {input_file} &")
        
        with open('run_pf_sims.bsh', 'w') as f:
            f.write('\n'.join(script_content))
        
        os.chmod('run_pf_sims.bsh', 0o755)
        print(f"Running batch of {len(batch_indices)} simulations...")
        subprocess.run(['bash', 'run_pf_sims.bsh'], check=True, capture_output=True)

    def run_simulations_with_params(self, sim_parameters):
        """
        Run simulations with pre-generated parameters
        
        Args:
            parameters: numpy array of shape (n_samples, 29) containing parameter sets
        """
        n_models = len(sim_parameters)
        failed = np.ones(n_models, dtype='bool')
        data = np.zeros((n_models, 65702))  # Adjust size if needed
        
        print(f"\nStarting simulation batch of {n_models} models...")
        print("=" * 50)
        
        while any(failed):
            failed_indices = np.where(failed)[0]
            
            self._run_parallel_simulations(failed_indices, sim_parameters)
            
            # Collect results
            for i in failed_indices:
                prefix = f"{self.simulator.output_prefix}_{i:04d}"
                try:
                    data[i] = self.ert_handler.gather_data(prefix)
                    failed[i] = False
                    print(f"Successfully completed simulation {i+1}/{n_models}")
                except:
                    print(f"WARNING: Simulation {i+1}/{n_models} failed and will be retried")
            
            # Print current progress
            success_count = n_models - np.sum(failed)
            print(f"\nProgress: {success_count}/{n_models} simulations completed successfully")
            print("=" * 50)
        
        print("\nAll simulations completed successfully!")
        return sim_parameters, data

    def run_simulations_with_params_single(self, sim_parameters, model_index):
        """
        Run simulations with pre-generated parameters in a single batch
        """
        script_content = ['#!/bin/bash']
        pflotran_cmd = f"mpirun -np 8 {self.pflotran_path} -pflotranin"
        
 
        input_file = self.simulator._create_input_file(sim_parameters, model_index)
        script_content.append(f"{pflotran_cmd} {input_file} &")
    
        with open('run_pf_sims.bsh', 'w') as f:
            f.write('\n'.join(script_content))
        
        os.chmod('run_pf_sims.bsh', 0o755)
        print(f"Running simulation number: {model_index} ...")
        subprocess.run(['bash', 'run_pf_sims.bsh'], check=True, capture_output=True)

  
        prefix = f"{self.simulator.output_prefix}_{model_index:04d}"
        
        try:
            sim_output = self.ert_handler.gather_data(prefix)
            print(f"Successfully completed simulation {model_index}")
        except:
            print(f"WARNING: Simulation {model_index} failed and will be retried")
    
        # Print current progress
        success_count = model_index
        print(f"\nProgress: {success_count} simulations completed successfully")
        print("=" * 50)

        return sim_output







class SurrogateDataGenerator:
    def __init__(self, param_limits):
        self.param_limits = param_limits
        
    def generate_training_samples(self, n_samples, method='lhs', plot_diagnostics=True):
        """
        Generate training samples using specified sampling method
        
        Args:
            n_samples: Number of samples to generate
            method: 'lhs' or 'sobol'
            plot_diagnostics: Whether to create diagnostic plots
        """
        if method == 'lhs':
            sampler = qmc.LatinHypercube(d=29)
        elif method == 'sobol':
            sampler = qmc.Sobol(d=29, scramble=True)
        else:
            raise ValueError("Method must be 'lhs' or 'sobol'")
            
        # Generate samples
        samples = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        parameters = np.zeros((n_samples, 29))
        for i in range(29):
            plims = self.param_limits.plims[i]
            if i in self.param_limits.logparms:
                log_min, log_max = np.log10(plims[0]), np.log10(plims[1])
                parameters[:,i] = 10**(log_min + (log_max - log_min)*samples[:,i])
            else:
                parameters[:,i] = plims[0] + (plims[1] - plims[0])*samples[:,i]
        
        if plot_diagnostics:
            self._plot_sampling_diagnostics(parameters)
            
        return parameters
    
    def _plot_sampling_diagnostics(self, parameters):
        """Create diagnostic plots for the sampling"""
        n_params = parameters.shape[1]
        
        # Create subplot grid
        n_plots = min(6, n_params)  # Show first 6 parameters
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Parameter Sampling Diagnostic Plots')
        
        for i in range(n_plots):
            row = i // 3
            col = i % 3
            
            # Histogram
            axs[row, col].hist(parameters[:,i], bins=30)
            axs[row, col].set_title(f'Parameter {i+1}')
            axs[row, col].set_xlabel('Value')
            axs[row, col].set_ylabel('Count')
            
        plt.tight_layout()
        plt.savefig('sampling_diagnostics.png')
        plt.close()
        
        # Create correlation plot
        plt.figure(figsize=(10, 10))
        correlation = np.corrcoef(parameters.T)
        plt.imshow(correlation, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Parameter Correlation Matrix')
        plt.xlabel('Parameter Index')
        plt.ylabel('Parameter Index')
        plt.savefig('parameter_correlations.png')
        plt.close()

