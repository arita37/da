### This for Linux Ubuntu 16.0 environnment  ################

##### Install of Conda env  #################################
bash
pwd

source activate base
conda create -n test01 intelpython3_full -c intel -y

source activate test01
pip install -r pip_freeze.txt



####  Coding Style ##########################################
2 possibilities :
   Functional programming or Oriented Object programming.
   For Data Processing , Functionnal programming is better (ie Spark Scala..) :
        Better Parallelism,
        Lower CPU overhead for data processing.
        Serialization (ie pickle) is safer/better.

   We use functionnal programming approach.




#### Data report model  #####################################





#### Feature analyis / Model training model  #################



####  Launch prediction model by CLI ########################











