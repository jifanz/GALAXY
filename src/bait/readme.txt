This code is based on both https://github.com/ej0cl6/deep-active-learning and https://github.com/JordanAsh/badge. The entry point is run.py, which allows the user to specify the active learning algorithm (alg), the number of samples to query at each round (nQuery), the number of labeled samples to start with (nStart), and other parameters. An experiment can be executed with a command like

python run.py --model resnet --nQuery 1000 --data SVHN --alg bait --nStart 100

which will select 1000 samples at a time from the SVHN dataset using the BAIT algorithm. The resnet will be retrained after every selection, and the program will not terminate until the entire dataset has been labeled.




 
