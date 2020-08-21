## VAE_text -Pytorch
Short overview:

Implementation of Varational Auto Encoder for text generation.
In this project I trained it over 10 seasons of freinds.
Created one model which encodes every character specific distribution. 
This was done by adding to the vocabulary a charecter token. (The same as <SOS> but there are severals <SOS>
each for every charecter)
for example in freinds the tokens are : [“Phoebe_speak”,”Joey_spaek”,” Ross_spaek” ,” Rachel_speak”,” Monica_spaek”,” Chandler _speak”]
 
Model architecture:
## simple Model architecture visualization

![](./Simple_mode_visual.PNG)




## Some results of generation:
### There severeal options of generation:
option 1 :  given a start word or a sentence,condition the rest of generation.
more specificly I inject to the decoder a charecter Token and the first word (or a sentence).
In order to generate the attention model weights the given distribution condition the input word(or sentence).

Results:
Generation of specific character: 
Give  a first word and :

the first word is : “sex”  
rachel:
['i', 'dont', 'think', 'that', 'this', 'cut', 'a', 'deal', 'you', 'know', 'i', 'have', 'to', 'say', 'i', 'have', 'a', 'lot', 'of', 'the', 'condom', '?', 'i', 'i', 'have']

['well', '!', 'i', 'mean', 'you', 'can', 'be', 'assured', '?', '!', 'this', 'is', 'a', 'good', 'person', '!', '!', 'you', 'just', 'do']

monica:
['oh', '!', 'oh', 'i', 'mean', 'i', 'know', 'what', 'i', 'was', 'just', 'going', 'out', 'to', 'me', '!', 'and', '!', 'you', 'mean', 'that', 'it', 'doesn\x92t', 'work', 'out']

joey:
['yeah', 'what', 'i', 'got', 'ta', 'be', 'so', 'awkward', 'i', 'have', 'to', 'do', 'a', 'baby', 'on', 'the', 'first', 'decision', 'you', 'can']

 
