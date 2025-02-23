import os
import time
import glob
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from cmdstanpy import CmdStanModel

from sccala.scmlib.models import SCM_Model
from sccala.utillib.aux import distmod_kin, quantile, split_list, nullify_output
from sccala.utillib.const import C_LIGHT


class SccalaSCM:
    def __init__(self, file, calib="CALIB", blind=True, blindkey="HUBBLE"):
        """
        Initializes SccalaSCM object.

        Parameters
        ----------
        file : str
            Path to file containing collected data for SCM.
        calib : str
            Identifier for calibrator sample dataset. All SNe containing
            this identifier in dataset name will be collected as calibrators.
            Default: "CALIB"
        blind : bool
            If True, H0 will not be revealed after the fit and saved in an
            encrypted format based on the 'blindkey'. Default: True
        blindkey : str
            Encryption key used for encrypting the H0 values before
            saving the output file.
        """

        if blind:
            assert blindkey is not None, (
                "For blinding, a blindkey has to be specified..."
            )
        self.blind = blind
        self.blindkey = blindkey

        df = pd.read_csv(file)

        datasets = list(df["dataset"].unique())

        if calib:
            calib_datasets = [x for x in datasets if calib in x]
            datasets = [x for x in datasets if calib not in x]
        else:
            calib_datasets = []

        self.sn = df[df["dataset"].isin(datasets)]["SN"].to_numpy()

        self.mag = df[df["dataset"].isin(datasets)]["mag"].to_numpy()
        self.mag_err = df[df["dataset"].isin(datasets)]["mag_err"].to_numpy()

        self.col = df[df["dataset"].isin(datasets)]["col"].to_numpy()
        self.col_err = df[df["dataset"].isin(datasets)]["col_err"].to_numpy()

        self.vel = df[df["dataset"].isin(datasets)]["vel"].to_numpy()
        self.vel_err = df[df["dataset"].isin(datasets)]["vel_err"].to_numpy()

        self.ae = df[df["dataset"].isin(datasets)]["ae"].to_numpy()
        self.ae_err = df[df["dataset"].isin(datasets)]["ae_err"].to_numpy()

        self.red = df[df["dataset"].isin(datasets)]["red"].to_numpy()
        self.red_err = df[df["dataset"].isin(datasets)]["red_err"].to_numpy()

        self.mag_sys = df[df["dataset"].isin(datasets)]["mag_sys"].to_numpy()
        self.v_sys = df[df["dataset"].isin(datasets)]["vel_sys"].to_numpy()
        self.c_sys = df[df["dataset"].isin(datasets)]["col_sys"].to_numpy()
        self.ae_sys = df[df["dataset"].isin(datasets)]["ae_sys"].to_numpy()

        self.epoch = df[df["dataset"].isin(datasets)]["epoch"].to_numpy()

        self.m_cut_nom = df[df["dataset"].isin(datasets)]["m_cut_nom"].to_numpy()
        self.sig_cut_nom = df[df["dataset"].isin(datasets)]["sig_cut_nom"].to_numpy()

        if calib:
            self.calib_sn = df[df["dataset"].isin(calib_datasets)]["SN"].to_numpy()

            self.calib_mag = df[df["dataset"].isin(calib_datasets)]["mag"].to_numpy()
            self.calib_mag_err = df[df["dataset"].isin(calib_datasets)][
                "mag_err"
            ].to_numpy()

            self.calib_col = df[df["dataset"].isin(calib_datasets)]["col"].to_numpy()
            self.calib_col_err = df[df["dataset"].isin(calib_datasets)][
                "col_err"
            ].to_numpy()

            self.calib_vel = df[df["dataset"].isin(calib_datasets)]["vel"].to_numpy()
            self.calib_vel_err = df[df["dataset"].isin(calib_datasets)][
                "vel_err"
            ].to_numpy()

            self.calib_ae = df[df["dataset"].isin(calib_datasets)]["ae"].to_numpy()
            self.calib_ae_err = df[df["dataset"].isin(calib_datasets)][
                "ae_err"
            ].to_numpy()

            self.calib_red = df[df["dataset"].isin(calib_datasets)]["red"].to_numpy()
            self.calib_red_err = df[df["dataset"].isin(calib_datasets)][
                "red_err"
            ].to_numpy()

            self.calib_mag_sys = df[df["dataset"].isin(calib_datasets)][
                "mag_sys"
            ].to_numpy()
            self.calib_v_sys = df[df["dataset"].isin(calib_datasets)][
                "vel_sys"
            ].to_numpy()
            self.calib_c_sys = df[df["dataset"].isin(calib_datasets)][
                "col_sys"
            ].to_numpy()
            self.calib_ae_sys = df[df["dataset"].isin(calib_datasets)][
                "ae_sys"
            ].to_numpy()

            self.calib_dist_mod = df[df["dataset"].isin(calib_datasets)][
                "mu"
            ].to_numpy()
            self.calib_dist_mod_err = df[df["dataset"].isin(calib_datasets)][
                "mu_err"
            ].to_numpy()

            self.calib_epoch = df[df["dataset"].isin(calib_datasets)][
                "epoch"
            ].to_numpy()

            self.calib_m_cut_nom = df[df["dataset"].isin(calib_datasets)][
                "m_cut_nom"
            ].to_numpy()
            self.calib_sig_cut_nom = df[df["dataset"].isin(calib_datasets)][
                "sig_cut_nom"
            ].to_numpy()
        else:
            self.calib_sn = None

            self.calib_mag = None
            self.calib_mag_err = None

            self.calib_col = None
            self.calib_col_err = None

            self.calib_vel = None
            self.calib_vel_err = None

            self.calib_ae = None
            self.calib_ae_err = None

            self.calib_red = None
            self.calib_red_err = None

            self.calib_mag_sys = None
            self.calib_v_sys = None
            self.calib_c_sys = None
            self.calib_ae_sys = None

            self.calib_dist_mod = None
            self.calib_dist_mod_err = None

            self.calib_epoch = None

            self.calib_m_cut_nom = None
            self.calib_sig_cut_nom = None

        self.datasets = df[df["dataset"].isin(datasets)]["dataset"].to_numpy()

        if calib:
            self.calib_datasets = df[df["dataset"].isin(calib_datasets)][
                "dataset"
            ].to_numpy()
        else:
            self.calib_datasets = None
        self.posterior = None

        return

    # TODO various methods to modify (add/ delete SNe) and display loaded data

    @property
    def red_uncertainty(self):
        red_uncertainty = (
            (
                self.red_err
                * 5
                * (1 + self.red)
                / (self.red * (1 + 0.5 * self.red) * np.log(10))
            )
            ** 2
            + (
                300
                / C_LIGHT
                * 5
                * (1 + self.red)
                / (self.red * (1 + 0.5 * self.red) * np.log(10))
            )
            ** 2
            + (0.055 * self.red) ** 2
        )
        return red_uncertainty

    def get_error_matrix(self, classic=False, rho=1.0, rho_calib=0.0):
        errors = []
        if not classic:
            for i in range(len(self.mag)):
                errors.append(
                    np.array([
                        [
                            self.red_uncertainty[i]
                            + self.mag_err[i] ** 2
                            + self.mag_sys[i] ** 2,
                            0,
                            self.mag_err[i] * self.col_err[i] * rho,
                            0,
                        ],
                        [0, self.vel_err[i] ** 2 + self.v_sys[i] ** 2, 0, 0],
                        [
                            self.mag_err[i] * self.col_err[i] * rho,
                            0,
                            self.col_err[i] ** 2 + self.c_sys[i] ** 2,
                            0,
                        ],
                        [0, 0, 0, self.ae_err[i] ** 2 + self.ae_sys[i] ** 2],
                    ])
                )
        else:
            for i in range(len(self.mag)):
                errors.append(
                    np.array([
                        [
                            self.red_uncertainty[i]
                            + self.mag_err[i] ** 2
                            + self.mag_sys[i] ** 2,
                            0,
                            self.mag_err[i] * self.col_err[i] * rho,
                        ],
                        [0, self.vel_err[i] ** 2 + self.v_sys[i] ** 2, 0],
                        [
                            self.mag_err[i] * self.col_err[i] * rho,
                            0,
                            self.col_err[i] ** 2 + self.c_sys[i] ** 2,
                        ],
                    ])
                )
        return np.array(errors)

    def sample(
        self,
        model,
        log_dir="log_dir",
        chains=4,
        iters=1000,
        warmup=1000,
        rho=1.0,
        rho_calib=0.0,
        save_warmup=False,
        quiet=False,
        init=None,
        classic=False,
        output_dir=None,
        test_data=False,
        selection_effects=True,
    ):
        """
        Samples the posterior for the given data and model using
        the cmdstanpy interface.

        Parameters
        ----------
        model : SCM_Model
            Model for which to fit the data
        log_dir : str
            Directory in which to save sampling output. If None is passed,
            result will not be saved. Default: log_dir
        chains : int
            Number of chains used in STAN fit. Default: 4
        iters : int
            Number of iterations used in STAN fit. Default: 1000
        warmup : int
            Number of iterations used in STAN fit as warmup. Default: 1000
        rho : float
            Correlation between the color and magnitude uncertainties. Default: 1.0
        rho_calib : float
            Correlation between the color and magnitude uncertainties for calibrator SNe. Default: 0.0
        save_warmup : bool
            If True, warmup elements of chain will be saved as well. Default: False
        quiet : bool
            Enables/ disables output statements after sampling has
            finished. Default: False
        classic : bool
            Switches classic mode on if True. In classic mode, a/e input is
            ignored.
        output_dir : str
            Directory where temporary STAN files will be stored. Default: None
        test_data : bool
            If True, the normalisation of the data will be overwritten to account
            for the test data. For now the values can't be adjusted and are
            hardcoded to the defaults of the `gen_testdata` script.
            Default: False
        selection_effects : bool
            If True, selection effects are included in the model. Default: True

        Returns
        -------
        posterior : pandas DataFrame

        """

        assert issubclass(type(model), SCM_Model), (
            "'model' should be a subclass of SCM_Model"
        )
        assert isinstance(iters, int), "'iters' has to by of type 'int'"
        assert isinstance(warmup, int), "'warmup' has to by of type 'int'"

        errors = self.get_error_matrix(classic=classic, rho=rho, rho_calib=rho_calib)

        if not classic:
            obs = np.array([self.mag, self.vel, self.col, self.ae]).T
            model.data["ae_avg"] = np.mean(self.ae)
        else:
            obs = np.array([self.mag, self.vel, self.col]).T

        # Fill model data
        model.data["sn_idx"] = len(self.sn)
        model.data["obs"] = obs
        model.data["errors"] = errors
        model.data["vel_avg"] = np.mean(self.vel)
        model.data["col_avg"] = np.mean(self.col)
        model.data["log_dist_mod"] = np.log10(distmod_kin(self.red))

        # For now, we take the average of the limiting magnitudes
        if selection_effects:
            model.data["m_cut_nom"] = np.mean(self.m_cut_nom)
            model.data["sig_cut_nom"] = np.mean(self.sig_cut_nom)
            model.data["use_selection"] = 1
        else:
            model.data["m_cut_nom"] = 0
            model.data["sig_cut_nom"] = 0
            model.data["use_selection"] = 0

        if test_data:
            model.data["vel_avg"] = 7100e3
            model.data["col_avg"] = 0.5
            if not classic:
                model.data["ae_avg"] = 0.31

        if model.hubble:
            assert self.calib_sn is not None, "No calibrator SNe found..."

            if not classic:
                calib_obs = np.array([
                    self.calib_mag,
                    self.calib_vel,
                    self.calib_col,
                    self.calib_ae,
                ]).T

                # Redshift, peculiar velocity and gravitational lensing uncertaintes
                calib_errors = np.array([
                    self.calib_mag_err**2
                    + self.calib_dist_mod_err**2
                    + self.calib_mag_sys**2,
                    self.calib_vel_err**2 + self.calib_v_sys**2,
                    self.calib_col_err**2 + self.calib_c_sys**2,
                    self.calib_ae_err**2 + self.calib_ae_sys**2,
                ]).T
            else:
                # Observed values
                calib_obs = np.array([self.calib_mag, self.calib_vel, self.calib_col]).T

                # Redshift, peculiar velocity and gravitational lensing uncertaintes
                calib_errors = np.array([
                    self.calib_mag_err**2
                    + self.calib_dist_mod_err**2
                    + self.calib_mag_sys**2,
                    self.calib_vel_err**2 + self.calib_v_sys**2,
                    self.calib_col_err**2 + self.calib_c_sys**2,
                ]).T
            model.data["calib_sn_idx"] = len(self.calib_sn)
            model.data["calib_obs"] = calib_obs
            model.data["calib_errors"] = calib_errors
            model.data["calib_dist_mod"] = self.calib_dist_mod

            # Convert differnet datasets to dataset indices
            n_calib_dset = len(set(self.calib_datasets))
            mappded_dsets = dict(
                zip(sorted(set(self.calib_datasets)), range(n_calib_dset))
            )
            # Plus one to take care of 1-based stan indexing
            calib_dset_idx = map(lambda x: mappded_dsets[x] + 1, self.calib_datasets)
            model.data["calib_dset_idx"] = list(calib_dset_idx)
            model.data["num_calib_dset"] = n_calib_dset

            # For now, we take the average for each dataset
            if selection_effects:
                calib_m_cut_nom = []
                calib_sig_cut_nom = []
                for i in range(n_calib_dset):
                    mask = [x == (i + 1) for x in model.data["calib_dset_idx"]]
                    calib_m_cut_nom.append(np.mean(self.calib_m_cut_nom[mask]))
                    calib_sig_cut_nom.append(np.mean(self.calib_sig_cut_nom[mask]))
                model.data["calib_m_cut_nom"] = np.array(calib_m_cut_nom)
                model.data["calib_sig_cut_nom"] = np.array(calib_sig_cut_nom)
                model.data["use_selection"] = 1
            else:
                model.data["calib_m_cut_nom"] = np.zeros(n_calib_dset)
                model.data["calib_sig_cut_nom"] = np.zeros(n_calib_dset)
                model.data["use_selection"] = 0

        model.set_initial_conditions(init)

        if log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        data_file = model.write_json("data.json", path=log_dir)

        mdl = CmdStanModel(stan_file=model.file)

        fit = mdl.sample(
            data=data_file,
            chains=chains,
            iter_warmup=warmup,
            iter_sampling=iters,
            save_warmup=save_warmup,
            inits=[model.init] * chains,
            output_dir=output_dir,
        )

        summary = fit.summary()
        diagnose = fit.diagnose()

        if not quiet:
            print(summary)
            print(diagnose)

        self.posterior = fit.draws_pd()

        # Encrypt H0 for blinding
        if self.blind and model.hubble:
            from itsdangerous import URLSafeSerializer

            norm = np.mean(self.posterior["H0"])
            s = URLSafeSerializer(self.blindkey)
            for i in range(len(self.posterior["H0"])):
                self.posterior["H0"][i] = s.dumps(self.posterior["H0"][i] / norm)
            norm = s.dumps(norm)
        else:
            norm = None

        if log_dir is not None:
            savename = self.__save_samples__(self.posterior, log_dir=log_dir, norm=norm)
            chains_dir = savename.replace(".csv", "")
            os.makedirs(chains_dir)
            with open(os.path.join(chains_dir, "summary.txt"), "w") as f:
                f.write(summary.to_string())
            with open(os.path.join(chains_dir, "diagnose.txt"), "w") as f:
                f.write(diagnose)
            if not self.blind:
                # Only move the csv files if we're not blinding the result
                # TODO: find a way of blinding the individual chains
                fit.save_csvfiles(chains_dir)

        if not quiet:
            model.print_results(self.posterior, blind=self.blind)

        return self.posterior

    def bootstrap(
        self,
        model,
        output,
        log_dir="log_dir",
        chains=2,
        iters=1000,
        warmup=1000,
        rho=1.0,
        rho_calib=0.0,
        save_warmup=False,
        save_chains=False,
        init=None,
        classic=False,
        replacement=True,
        restart=True,
        walltime=24.0,
        output_dir=None,
        selection_effects=True,
    ):
        """
        Samples the posterior for the given data and model

        Parameters
        ----------
        model : SCM_Model
            Model for which to fit the data
        output : str
            Name of the file where the resampled H0 values should be
            written to.
        log_dir : str
            Directory in which to save sampling output. If None is passed,
            result will not be saved. Default: log_dir
        chains : int
            Number of chains used in STAN fit. Default: 4
        iters : int
            Number of iterations used in STAN fit. Default: 1000
        warmup : int
            Number of iterations used in STAN fit as warmup. Default: 1000
        save_warmup : bool
            If True, warmup elements of chain will be saved as well. Default: False
        save_chains : bool
            If True, individual chains of each bootstrap step will be saved.
            WARNING: This will use huge amounts of disk space! Default: False
        classic : bool
            Switches classic mode on if True. In classic mode, a/e input is
            ignored.
        replacement : bool
            If True, bootstrap resampling will be done with replacement.
            Default: True
        restart : bool
            Enables writing of restart files. Can be disabled if disk space is a
            concern. Restart files will be written in the log_dir. Default: True
        time : float
            Wallclock time (in h) available. Once 95% of the available wallclock
            time is used, no new iteration will be started and job will exit
            cleanly. Should be used with restart set to True. Default 24.0
        output_dir : str
            Directory where temporary STAN files will be stored. Default: None
        selection_effects : bool
            If True, selection effects are included in the model. Default: True

        Returns
        -------
        bootstrap_h0 : list of float
            List containing 50th quantiles of each individual bootstrap step.
        """

        start = time.clock_gettime(time.CLOCK_REALTIME)

        assert issubclass(type(model), SCM_Model), (
            "'model' should be a subclass of SCM_Model"
        )
        assert isinstance(iters, int), "'iters' has to by of type 'int'"
        assert isinstance(warmup, int), "'warmup' has to by of type 'int'"
        assert self.calib_sn is not None, "There are no calibrator SNe"
        assert model.hubble, "Bootstrap resampling only works for H0 models"
        if restart:
            assert log_dir is not None, "Restart required valid log_dir"

        # Some checks to make sure that the chain length etc. aren't too long.
        if chains > 2:
            warnings.warn(
                "High number of chains detected (%d), this might thake a while..."
                % chains
            )
        if iters > 2000:
            warnings.warn(
                "High number of iterations detected (%d), this might thake a while..."
                % iters
            )
        if warmup > 2000:
            warnings.warn(
                "High number of warmup iterations detected (%d), this might thake a while..."
                % warmup
            )
        if save_warmup or save_chains:
            warnings.warn(
                "Saving of chains enabled. This will use HUGE amounts of disk space!"
            )

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            parallel = True
        except ModuleNotFoundError:
            comm = None
            rank = 0
            size = 1
            parallel = False

        if log_dir is not None and rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        errors = self.get_error_matrix(classic=classic, rho=rho)

        if not classic:
            obs = np.array([self.mag, self.vel, self.col, self.ae]).T
            model.data["ae_avg"] = np.mean(self.ae)
        else:
            # Observed values
            obs = np.array([self.mag, self.vel, self.col]).T

        # Fill model data
        model.data["sn_idx"] = len(self.sn)
        model.data["obs"] = obs
        model.data["errors"] = errors
        model.data["vel_avg"] = np.mean(self.vel)
        model.data["col_avg"] = np.mean(self.col)
        model.data["log_dist_mod"] = np.log10(distmod_kin(self.red))

        if selection_effects:
            model.data["m_cut_nom"] = np.mean(self.m_cut_nom)
            model.data["sig_cut_nom"] = np.mean(self.sig_cut_nom)
            model.data["use_selection"] = 1
        else:
            model.data["m_cut_nom"] = 0
            model.data["sig_cut_nom"] = 0
            model.data["use_selection"] = 0

        if not classic:
            calib_obs = np.array([
                self.calib_mag,
                self.calib_vel,
                self.calib_col,
                self.calib_ae,
            ]).T

            # Redshift, peculiar velocity and gravitational lensing uncertaintes
            calib_errors = np.array([
                self.calib_mag_err**2
                + self.calib_dist_mod_err**2
                + self.calib_mag_sys**2,
                self.calib_vel_err**2 + self.calib_v_sys**2,
                self.calib_col_err**2 + self.calib_c_sys**2,
                self.calib_ae_err**2 + self.calib_ae_sys**2,
            ]).T
        else:
            calib_obs = np.array([self.calib_mag, self.calib_vel, self.calib_col]).T

            # Redshift, peculiar velocity and gravitational lensing uncertaintes
            calib_errors = np.array([
                self.calib_mag_err**2
                + self.calib_dist_mod_err**2
                + self.calib_mag_sys**2,
                self.calib_vel_err**2 + self.calib_v_sys**2,
                self.calib_col_err**2 + self.calib_c_sys**2,
            ]).T

        # Generate list with all bootstrap combinations.
        indices = np.arange(len(self.calib_sn))
        if replacement:
            bt_inds = list(
                itertools.combinations_with_replacement(indices, len(self.calib_sn))
            )
        else:
            bt_inds = list(itertools.combinations(indices, len(self.calib_sn)))

        h0_vals = []

        if rank == 0:
            print(
                "Beginning bootstrap resampling for %d combinations on %d ranks, this might take a while..."
                % (len(bt_inds), size)
            )

        # Check if restart files exist for each rank
        restart_files = glob.glob("restart_*.dat", root_dir=log_dir)
        if len(restart_files) == 0 or not restart:
            if restart:
                restart_files = ["restart_%03d.dat" % i for i in range(size)]
            found_restart = False
        else:
            assert len(restart_files) == size, (
                "Mismatch between number of restart files (%d) and ranks (%d)"
                % (
                    len(restart_files),
                    size,
                )
            )
            restart_files.sort()
            found_restart = True

        # Some parallelization stuff
        if parallel:
            perrank = int(np.ceil(len(bt_inds) / size))
            bt_inds_lists = list(split_list(bt_inds, perrank))
            tr = trange(len(bt_inds_lists[rank]), desc="Rank %d" % rank, position=rank)
        else:
            perrank = len(bt_inds)
            bt_inds_lists = bt_inds
            tr = trange(len(bt_inds_lists), desc="Rank %d" % rank, position=rank)

        if found_restart:
            done = np.genfromtxt(os.path.join(log_dir, restart_files[rank]), dtype=int)
        else:
            done = []

        if rank == 0:
            # Create a model instance to trigger compilation and avoid
            # having to compile the model on each rank separately
            print("Compiling model...")
            mdl_0 = CmdStanModel(stan_file=model.file)
            del mdl_0
            print("Model compiled, starting sampling...")

        comm.Barrier()

        if output_dir is not None:
            output_dir_rank = os.path.join(output_dir, "rank_%03d" % rank)
            if not os.path.exists(output_dir_rank):
                os.makedirs(output_dir_rank)
        else:
            output_dir_rank = None

        comm.Barrier()

        for k in tr:
            if parallel:
                inds = bt_inds_lists[rank][k]
            else:
                inds = bt_inds_lists[k]

            # Check if index combination has already been done
            if found_restart and not parallel:
                if any(np.equal(done, inds).all(1)):
                    continue
            elif found_restart and parallel:
                try:
                    if any(np.equal(done, inds).all(1)):
                        continue
                except np.AxisError:
                    if any(np.equal(done, inds)):
                        continue

            model.data["calib_sn_idx"] = len(self.calib_sn)
            model.data["calib_obs"] = np.array([calib_obs[i] for i in inds])
            model.data["calib_errors"] = np.array([calib_errors[i] for i in inds])
            model.data["calib_dist_mod"] = np.array([
                self.calib_dist_mod[i] for i in inds
            ])

            # Convert differnet datasets to dataset indices
            active_datasets = [self.calib_datasets[i] for i in inds]
            n_calib_dset = len(set(active_datasets))
            mappded_dsets = dict(zip(sorted(set(active_datasets)), range(n_calib_dset)))
            # Plus one to take care of 1-based stan indexing
            calib_dset_idx = map(lambda x: mappded_dsets[x] + 1, active_datasets)

            model.data["calib_dset_idx"] = list(calib_dset_idx)
            model.data["num_calib_dset"] = n_calib_dset

            model.set_initial_conditions(init)

            # Setup/ build STAN model
            with nullify_output(suppress_stdout=True, suppress_stderr=True):
                data_file = model.write_json(f"data_{rank}.json", path=log_dir)

                mdl = CmdStanModel(stan_file=model.file)

                fit = mdl.sample(
                    data=data_file,
                    chains=chains,
                    iter_warmup=warmup,
                    iter_sampling=iters,
                    save_warmup=save_warmup,
                    inits=[model.init] * chains,
                    output_dir=output_dir_rank,
                )

                self.posterior = fit.draws_pd()

            # Append found H0 values to list
            h0 = quantile(self.posterior["H0"], 0.5)
            h0_vals.append(h0)

            if log_dir is not None and save_chains:
                self.__save_samples__(self.posterior, log_dir=log_dir, norm=None)

            # Save resampled h0 values
            if not parallel:
                with open(output, "a") as f:
                    f.write("%g\n" % h0)
                # Append index list to restart file
                if restart:
                    with open(os.path.join(log_dir, restart_files[rank]), "a") as f:
                        done_inds = " ".join(map(str, inds)) + "\n"
                        f.write(done_inds)
            else:
                with open("output_%03d.tmp" % rank, "a") as f:
                    f.write("%g\n" % h0)
                if restart:
                    with open(os.path.join(log_dir, restart_files[rank]), "a") as f:
                        done_inds = " ".join(map(str, inds)) + "\n"
                        f.write(done_inds)

            # Time passed in h
            time_passed = (time.clock_gettime(time.CLOCK_REALTIME) - start) / 3600
            if time_passed > 0.95 * walltime:
                print("[TIMELIMIT] Rank %d reached wallclock limit, exiting..." % rank)
                break

            # If not using the default output_dir, clean up the temporary files to avoid
            # excessive disk usage
            if output_dir is not None:
                files = glob.glob(os.path.join(output_dir_rank, "*"))
                for f in files:
                    os.remove(f)

        if parallel:
            comm.Barrier()
            h0_vals = comm.gather(h0_vals, root=0)
            if rank == 0:
                h0_vals = [item for sublist in h0_vals for item in sublist]
                if parallel:
                    # Collect all H0 values into one h0_file
                    with open(output, "ab") as f:
                        np.savetxt(f, h0_vals)

        return h0_vals

    def __save_samples__(self, df, log_dir="log_dir", norm=None):
        """Exports sample data"""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        savename = "chains_1.csv"
        if os.path.exists(os.path.join(log_dir, savename)):
            i = 1
            while os.path.exists(
                os.path.join(log_dir, savename.replace("1", str(i + 1)))
            ):
                i += 1
            savename = savename.replace("1", str(i + 1))

        df.to_csv(os.path.join(log_dir, savename))

        if norm is not None:
            normsave = savename.replace(".csv", ".key")
            np.savetxt(os.path.join(log_dir, normsave), [norm], fmt="%s")

        return os.path.join(log_dir, savename)

    def hubble_diagram(self, save=None, classic=False):
        """
        Plots a Hubble diagram of the posterior

        Parameters
        ----------
        save : str
            Specified where the generated Hubble diagram will be saved.
        classic : bool
            Switches classic mode on if True. In classic mode, a/e input is
            ignored.

        Returns
        -------
        None

        Note: Only plots redshift vs. apparent magnitude to work for all
        models, i.e. H0 and H0-free models.
        """

        if self.posterior is None:
            warnings.warn("Please run sampling before generating Hubble diagram")
            return

        mi = self.posterior["Mi"].to_numpy().mean()
        alpha = self.posterior["alpha"].to_numpy().mean()
        beta = self.posterior["beta"].to_numpy().mean()
        try:
            h0 = self.posterior["H0"].to_numpy().mean()
            hubble = True
        except KeyError:
            h0 = None
            hubble = False

        m_corr = (
            self.mag
            + alpha * np.log10(self.vel / np.mean(self.vel))
            - beta * (self.col - np.mean(self.col))
        )
        if hubble:
            calib_m_corr = (
                self.calib_mag
                + alpha * np.log10(self.calib_vel / np.mean(self.vel))
                - beta * (self.calib_col - np.mean(self.col))
            )
        if not classic:
            gamma = self.posterior["gamma"].to_numpy().mean()
            m_corr -= gamma * (self.ae - np.mean(self.ae))
            if hubble:
                calib_m_corr -= gamma * (self.calib_ae - np.mean(self.ae))

        res = 5 * np.log10(distmod_kin(self.red)) + mi - self.mag
        res_corr = 5 * np.log10(distmod_kin(self.red)) + mi - m_corr
        if hubble:
            res -= 5 * np.log10(h0) - 25
            res_corr -= 5 * np.log10(h0) - 25
            res_calib = (
                mi
                - self.calib_mag
                + 5 * np.log10(distmod_kin(self.calib_red))
                - 5 * np.log10(h0)
                + 25
            )
            res_calib_corr = (
                mi
                - calib_m_corr
                + 5 * np.log10(distmod_kin(self.calib_red))
                - 5 * np.log10(h0)
                + 25
            )

        fig, ax = plt.subplots()
        fig = plt.figure(figsize=[10.2, 7.2])
        ax = fig.add_axes((0.1, 0.3, 0.8, 0.6))
        ax.scatter(
            self.red, self.mag, color="tab:blue", alpha=0.5, label=r"$m_\mathrm{obs}$"
        )
        ax.errorbar(
            self.red,
            m_corr,
            yerr=self.mag_err,
            fmt="o",
            color="tab:blue",
            label=r"$m_\mathrm{corr}$",
        )
        if hubble:
            red_min = np.min(np.concatenate([self.red, self.calib_red]))
            red_max = np.max(np.concatenate([self.red, self.calib_red]))
        else:
            red_min = np.min(self.red)
            red_max = np.max(self.red)
        x = np.linspace(red_min, red_max, 100)
        cosmo = 5 * np.log10(distmod_kin(x)) + mi
        if hubble:
            cosmo -= 5 * np.log10(h0) - 25
            ax.scatter(
                self.calib_red,
                self.calib_mag,
                color="tab:orange",
                alpha=0.5,
                label=r"$m_\mathrm{calib}$",
            )
            ax.errorbar(
                self.calib_red,
                calib_m_corr,
                yerr=self.calib_mag_err,
                fmt="o",
                color="tab:orange",
                label=r"$m_\mathrm{calib, corr}$",
            )
        ax.plot(x, cosmo, color="k", ls="--", label="Cosmology")

        ax.set_ylabel(r"$m$ (mag)")
        ax.legend()

        ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.2))
        ax2.scatter(self.red, res, color="tab:blue", alpha=0.5)
        ax2.errorbar(
            self.red,
            res_corr,
            fmt="o",
            color="tab:blue",
            linewidth=2,
        )
        if hubble:
            ax2.scatter(self.calib_red, res_calib, color="tab:orange", alpha=0.5)
            ax2.errorbar(
                self.calib_red,
                res_calib_corr,
                fmt="o",
                color="tab:orange",
                linewidth=2,
            )

        plt.hlines(0, red_min, red_max, linestyles="--", color="k")

        ax2.set_xlabel(r"$z_\mathrm{CMB}$")
        ax2.set_ylabel("Residuals")
        plt.grid()

        fig.savefig(save, bbox_inches="tight", dpi=300)

    def cornerplot(self, save=None, classic=False):
        """
        Plots the cornerplot of the posterior

        Parameters
        ----------
        save : str
            Specified where the generated cornerplot will be saved.
        classic : bool
            Switches classic mode on if True. In classic mode, a/e input is
            ignored.

        Returns
        -------
        None
        """

        if self.posterior is None:
            warnings.warn("Please run sampling before generating cornerplots")
            return

        try:
            import corner
        except ImportError:
            warnings.warn("corner package not installed, skipping...")
            return

        if not classic:
            paramnames = [
                r"$\mathcal{M}_I$",
                r"$\alpha$",
                r"$\beta$",
                r"$\gamma$",
                r"$\sigma_{int}$",
            ]
            ndim = len(paramnames)

            # Get relevant parameters
            keys = ["Mi", "alpha", "beta", "gamma", "sigma_int"]
        else:
            paramnames = [
                r"$\mathcal{M}_I$",
                r"$\alpha$",
                r"$\beta$",
                r"$\sigma_{int}$",
            ]
            ndim = len(paramnames)

            # Get relevant parameters
            keys = ["Mi", "alpha", "beta", "sigma_int"]

        all_keys = list(self.posterior.keys())
        if "mag_cut" in all_keys:
            paramnames.append(r"$m_\mathrm{cut}$")
            keys.append("mag_cut")
            ndim += 1
        if "sigma_cut" in all_keys:
            paramnames.append(r"$\sigma_\mathrm{cut}$")
            keys.append("sigma_cut")
            ndim += 1

        posterior = self.posterior[keys].to_numpy()

        figure = corner.corner(
            posterior,
            labels=paramnames,
            show_titles=True,
        )

        # This is the empirical mean of the sample:
        value = np.mean(posterior, axis=0)

        # Extract the axes
        axes = np.array(figure.axes).reshape((ndim, ndim))

        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(value[i], color="g")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(value[xi], color="g")
                ax.axhline(value[yi], color="g")
                ax.plot(value[xi], value[yi], "sg")

        if isinstance(save, str):
            plt.savefig(
                save,
                bbox_inches="tight",
                dpi=300,
            )

        plt.close()

        return
