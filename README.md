# CycleGAN on Text Sentiment Transfer
  Unsupervised learning sentiment transfer from negative to positive and vice versa.  

## Implementation
* I used improved WGAN to conduct adversarial training.
* The two generators are pretrained by auto-encoder.
* The generators directly generate a word embedding at each time step, instead of generating word distribution.
* During testing, at each time step, I choose the word with maximum cosine similarity between generated word embedding as generated word.

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

## Acknoledgement
  The discriminator part of code I used can be found at:  
  `https://github.com/igul222/improved_wgan_training`  
