System requirements
-------------------
- Operating System

	UNIX clones such as LINUX and OS X.  MS Windows is not supported.

- [PETSc](http://www.mcs.anl.gov/petsc)

- [MaxwellFDFD](https://github.com/wsshin/maxwellfdfd)

	MaxwellFDFD is a companion MATLAB package to create the input files for FD3D and analyze the solution files.  MaxwellFDFD (and MATLAB) does not have to be installed in the same machine you install FD3D.  In fact, FD3D is typically installed on a LINUX cluster, whereas MaxwellFDFD is installed your local machine, such as your laptop.


PETSc installation instruction
------------------------------
1. Download the PETSc library source code from the [PETSc download page](http://www.mcs.anl.gov/petsc/petsc-as/download/index.html).

2. Unarchive the PETSc package, and move it to the directory where you want to install the library.

	For example, you can create a directory `petsc` under your home directory and move the package there.  Then, the contents of the PETSc package will be at `$HOME/petsc/petsc-X.Y.Z/`, where `X.Y.Z` is the version number.

3. Set up the environment variables `PETSC_DIR` and `PETSC_ARCH`.

	If your LINUX SHELL is `bash`, you can open `.bash_profile` file in your home directory (or create one if there is not), and add the following two lines to the file:

		export PETSC_DIR=YOUR_PETSC_DIRECTORY
		export PETSC_ARCH=petscfd3d

	where `YOUR_PETSC_DIRECTORY` is where your PETSc package is, e.g., `$HOME/petsc/petsc-X.Y.Z` in Step 2.

4. Make the environment variables effective.

	If your LINUX shell is bash and you followed Step 3, this can be done by executing:

		source `$HOME/.bash_profile`.

5. Copy the PETSc configuration script from FD3D.

	In your FD3D directory, you will find `petscfd3d.py` script.  Copy this to `$PETSC_DIR/config/` directory.

6. Make the configuration script executable.

	This can be done by executing:

		cd $PETSC_DIR/config
		chmod u+x petscfd3d.py

7. Configure PETSc.

	Execute:

		cd $PETSC_DIR
		config/petscfd3d.py

	Now, you will see a long configuration message.

8. Build PETSc.

	Execute:

		make all

9. Test PETSc.

	Execute:

		make test

	If you do not get any error, your PETSc installation is successful.  If you have any problem, you may find a solution at [PETSc Documentation: Installation](http://www.mcs.anl.gov/petsc/petsc-as/documentation/installation.html).


FD3D installation instruction
-----------------------------
Follow this instruction once you install the PETSc library following the instruction above.

1. Unarchive the FD3D package, and move it to the directory where you want to install the library.

	For example, you can put the unarchived package in your home directory.  Then the contents of the FD3D package will be at `$HOME/fd3d/`.

2.  Assign `FD3D_ROOT` environment variable.

	You have to let FD3D know its install location via an environment variable named `FD3D_ROOT`.  If your LINUX shell is bash, you can open `$HOME/.fd3d` file with your favorite text editor to append:

		setenv FD3D_ROOT $HOME/fd3d

	Then open `$HOME/.bash_profile` file with your favorite text editor to append:

		source $HOME/.fd3d
		export PATH=$PATH:$FD3D_ROOT/bin

3. Make the environment variable effective.

	Execute:

		source $HOME/.bash_profile

4. Build FD3D.

	Move to `$FD3D_ROOT/src/` directory by:

		cd $FD3D_ROOT/src

	If this does not place you in `$FD3D_ROOT/src/` directory, Step 1, 2 are not done correctly.

	Now, copy `$FD3D_ROOT/src/makefiles/makefile.tmp` file to `$FD3D_ROOT/src/` directory and rename it to makefile by:

		cp makefiles/makefile.tmp makefile

	Finally, build fd3d by executing:

		make

	in `$FD3D_ROOT/src/` directory.


5. Test FD3D.

	In `$FD3D_ROOT/test/` directory, there are input files `pointsrc_2d.h5`, `poinstsrc_2d.eps.gz`, `pointsrc_2d.srcJ.gz`, `pointsrc_2d.srcM.gz`.  To create these files you need to install MaxwellFDFD, a companion MATLAB package of FD3D, but these input files are provided so that you can test FD3D without installing MaxwellFDFD.  

	To test FD3D, execute:

		cd $FD3D_ROOT/test
		gzip -d pointsrc_2d.{eps,srcJ,srcM}.gz
		fd3d -i pointsrc_2d

	If this generates normal standard output without error messages, your FD3D installation is successful.
