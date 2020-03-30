# General notes

There are two main locations for installing modules: the module configurations folder, and the module package builds folder.

### Modules configurations folder
Location: `/opt/modulefiles/`

This folder contains the recipe file with the following information:
 * Module description, including an explanation of how to use it.
 * The software category to which the module belongs to. 
 * Dependencies to load before activating this (like other modules)
 * Conflicting modules to unload as to prevent clashing with this one.
 * Paths and variables to use while configuring the module.
 * Paths and variables to set after the module has been activated. 

These configurations are handled in the following in-house Git repository: https://gitlab.psc.edu/installers/bridges_modules , for which you have to setup an SSH key and request access before modifying the content.

### Modules package builds folder
Location: `/opt/packages/`

This folder contains the source code and/or general installation for generating the binary files required for each module to work. 
It also contains the configuration used for compiling modules, when applicable.

This folder contains files such as:
* Binary files.
* Libraries.
* Configure and Makefiles used for building the binary files.
* Conda files for environments built using it as a backend.

## How to create a new module.

The main ways include:

* Source code compilation: download the source code, set library paths, run configure files and set path prefixes, run the makefile. 
* Create Python environment: download a specific python version, set paths, install pip, install packages.
* Create Conda environment: load anaconda, create a new environment while specifying the path prefix to use, install packages.   


# WIP
    Now, when configuring a new module, the following process should take place:
    
    
    Inspect the previous module configuration
    
        cd /opt/packages/${MODULE_NAME}/${YEAR}/${SPECIFIC_MODULE_VERSION}/
        # Example:
        #   cd /opt/packages/caffe/git/caffe_master/  # This path has something else in the middle that might not be required ("git") and could use the YEAR field for clarity..
        
        # ls
        #   build           CONTRIBUTING.md  docs        Makefile                 Makefile.config_v1  python
        #   caffe.cloc      CONTRIBUTORS.md  examples    #Makefile#               make.log            README.md
        #   caffe_master    data             include     Makefile.config          matlab              scripts
        #   cmake           distribute       INSTALL.md  Makefile.config~         mnist_log.log       src
        #   CMakeLists.txt  docker           LICENSE     Makefile.config.example  models              tools
        
        # Inspect differences between the recipe for building the package and how it was actually built:
        #   diff Makefile.config Makefile.config.example >> /tmp/module.diff  
        #   vim /tmp/module.diff
        
        
        cd /opt/modulefiles/${MODULE_NAME}/${YEAR}/${SPECIFIC_MODULE_VERSION}/
        caffe train -solver=./solver_configuration.prototxt --weights ./bvlc_reference_caffenet.caffemodel
        
        caffe train -solver=./solver_configuration.prototxt --snapshot ./model_iter_11000.solverstate