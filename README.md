# CycleGAN on Text Sentiment Transfer
  Unsupervised learning sentiment transfer from negative to positive and vice versa.  

## Training
First pretrain generator by auto-encoder and create pretrai model:  
`$ python3 main.py -train -mode pretrain -model_dir 'your model path'`  
Load pretrain model and train cycleGAN:  
`$ python3 main.py -train -mode all -model_dir 'your model path'`

## Testing
  Requirement:  
  Tensorflow 1.2.1  

Run test:  
`$ python2 main.py -test -model_dir cur_best2`  


#### Examples
  i hate you->i love you  
  i can't do that-> i can do that  
  it's such a bad day-> it's such a good day  
  such a sad day->such a happy day  
  no it's not a good idea->it it ' s good idea  
