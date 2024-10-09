The files in this directory show an example vibrational density of states
(VDoS) calculation based on a molecular dynamics (MD) trajectory of an ethanol
system.

Contents:
	README.md
		the current README file
	equil_NVE1.data
		the initial data file for the ethanol system after equilibration in the
		microcanonical (NVE) ensemble
	vacf.in
		LAMMPS input script to run MD on the initial system and output
		velocity data
	vacf.log
		the LAMMPS log for the MD trajectory
	velocities\*.dump
		velocity output data for different samples from the MD trajectory
	dump2VDoS.py
		script to compute the velocity autocorrelation function (VACF) and VDoS
		from the dump files
	wavenumber.npy
		file containing the NumPy array of VDoS wavenumbers \[cm^{-1}\]
	intensity.npy
		file containing the NumPy array of VDoS intensities \[dimensionless\]