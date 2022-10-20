Installation
============

.. important::
    The installation notes below are tested only on Linux.

Steps
-----

1. Download InSAR4SM
^^^^^^^^^^^^^^^^^^^^^
First you have to download InSAR4SM using the following command.

.. code-block:: bash

	git clone https://github.com/kleok/InSAR4SM.git

2. Create python environment for InSAR4SM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
InSAR4SM is written in Python3 and relies on several Python modules. You can install them by using `INSAR4SM_env.yml <https://github.com/kleok/INSAR4SM/blob/main/INSAR4SM_env.yml>`_.


3. Set environmental variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On GNU/Linux, append to :file:`.bashrc` file:

.. code-block:: bash

    export InSAR4SM_HOME=~/FLOMPY
    export PYTHONPATH=${PYTHONPATH}:${InSAR4SM_HOME}
    export PATH=${PATH}:${InSAR4SM_HOME}
