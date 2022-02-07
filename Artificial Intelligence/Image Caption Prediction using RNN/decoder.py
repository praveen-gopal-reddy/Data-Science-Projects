#!/usr/bin/env python
# coding: utf-8

# In[15]:


"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""
import torch
import numpy as np
import argparse
from vocabulary import Vocabulary
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from datetime import datetime as dt
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity


from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *



# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True


# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)
print("Number of words in vocab:", vocab.idx)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


if not EVAL:

    # load the features saved from extract_features.py
    print("lines:",len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=2, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print("length of imageids:",len(image_ids))
    print("length of captions:",len(cleaned_captions))
    print("features dimension:",features.shape)


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here


    # for each batch, prepare the targets using this torch.nn.utils.rnn function
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
	
    total_step = len(train_loader)
    log_step=10                         #display log after every 10 batches
    for epoch in range(NUM_EPOCHS):
        for i,(features,captions,lengths) in enumerate(train_loader):
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            outputs = decoder(features,captions,lengths)
            loss=criterion(outputs,targets)
            decoder.zero_grad()        
            loss.backward()
            optimizer.step()
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, NUM_EPOCHS, i, total_step, loss.item(), np.exp(loss.item())))
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")
# if we already trained, and EVAL == True, reload saved model
else:

    data_transform = transforms.Compose([ 
        transforms.Resize(224),     
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])


    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)
    vocab_test = build_vocab_test(test_cleaned_captions)
    print("Number of words in vocab:", vocab_test.idx)


    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm
            


# In[ ]:


#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################


    # TODO define decode_caption() function in utils.py
    # predicted_caption = decode_caption(word_ids, vocab)
dataset_train_test = Flickr8k_Images(
    image_ids=test_image_ids,
    transform=data_transform,
)

train_loader_test = torch.utils.data.DataLoader(
    dataset_train_test,
    batch_size=1,
    shuffle=False,
    num_workers=2,
)
print(len(test_image_ids))
print(len(dataset_train_test))
print(len(train_loader_test))

for i,(test_images) in enumerate(train_loader_test):
    #test_images = test_images.to(device)
    f = encoder(test_images)
    sampled_ids = decoder.sample(f)
    sampled_ids = sampled_ids[0].cpu().numpy()
    listoflist=np.array(sampled_ids).tolist()
    predicted_cap=decode_caption(listoflist,vocab)    #calls decode_caption function
    caption_list=predicted_cap.split()
    caption_list = ' '.join(caption_list[1:len(caption_list)-1])
    print(caption_list)


#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report
    
#########1) BLEU for evaluation #########################################
vector_word_ids = []
all_test_ref=all_ref_pred
sum_score1=sum_score2=sum_score3=sum_score4=0
chencherry = SmoothingFunction()
high_score4=dict()
low_score4=dict()
bleu_score1=[]
bleu_score2=[]
bleu_score3=[]
bleu_score4=[]
bleu_scores_caption_imgid=dict()
n=0
for i,(test_images,image_id) in enumerate(train_loader_test):
    id_1=image_id[0]
    f = encoder(test_images)
    sampled_ids = decoder.sample(f)
    sampled_ids = sampled_ids[0].cpu().numpy()
    #print(sampled_ids)
    listoflist=np.array(sampled_ids).tolist()
    #print(listoflist)
    predicted_cap=decode_caption(listoflist,vocab)                  #calls decode_caption function
    #print(predicted_cap)
    caption_list=predicted_cap.split()
    #print(caption_list)
    caption_list = caption_list[1:len(caption_list)-1]
    reference=all_test_ref[id_1]["original"]
    score_1=sentence_bleu(reference,caption_list,weights=(1, 0, 0, 0),smoothing_function=chencherry.method4)
    score_2=sentence_bleu(reference,caption_list,weights=(0.5, 0.5, 0, 0),smoothing_function=chencherry.method4)
    score_3=sentence_bleu(reference,caption_list,weights=(0.33, 0.33, 0.33, 0),smoothing_function=chencherry.method4)
    score_4=sentence_bleu(reference,caption_list,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=chencherry.method4)
    bleu_score1.append(round(score_1,2))
    bleu_score2.append(round(score_2,2))
    bleu_score3.append(round(score_3,2))
    bleu_score4.append(round(score_4,2))
    score_4=round(score_4,2)
    bleu_scores_caption_imgid[id_1]=[score_4,' '.join(caption_list)]
    #find highest and lowest 4-gram score
    if score_4>0.8:
        high_score4[id_1]=score_4
    if score_4<0.1:
        low_score4[id_1]=score_4
    sum_score1+=score_1
    sum_score2+=score_2
    sum_score3+=score_3
    sum_score4+=score_4
    n+=1
    #print(n)
    #if i==100:
        #break
    #print(i)
average1=sum_score1/n
average2=sum_score2/n
average3=sum_score3/n
average4=sum_score4/n
print("average bleu-1-gram is {:.2f}:".format(round(average1,2)))
print("average bleu-2-gram is {:.2f}:".format(round(average2,2)))
print("average bleu-3-gram is {:.2f}:".format(round(average3,2)))
print("average bleu-4-gram is {:.2f}:".format(round(average4,2)))




######################(2) Cosine similarity####################

all_test_ref=all_ref_pred
cosine_scores_list=[]
cosine_scores_caption_imgid=dict()
cosine_rescale_caption_imgid=dict()
cosine_scores_rescale_list=[]
n=0
sum_score=0
sum_score_rescale=0
indx=0
for i,(test_images,image_id) in enumerate(train_loader_test):
    id_1=image_id[0]
    f = encoder(test_images)
    sampled_ids = decoder.sample(f)
    sampled_ids = sampled_ids[0].cpu().numpy()
    listoflist=np.array(sampled_ids).tolist()
    predicted_cap=decode_caption(listoflist,vocab)                     #calls decode_caption function
    #print(predicted_cap)
    caption_list=predicted_cap.split()
    #print(caption_list)
    caption_list = caption_list[1:len(caption_list)-1]
    #print("caption----", caption_list)
    caption_avg_vector = calculate_avg_vector(caption_list)
    reference=all_test_ref[id_1]["original"][indx]                
    #index for reference caption
    indx+=1
    if indx==5:
        indx=0
    #print("reference----", reference)
    reference_avg_vector = calculate_avg_vector(reference)
    cosine_result = cosine_similarity(reference_avg_vector,caption_avg_vector)      
    #print(cosine_result)
    cosine_scores_list.append(round(cosine_result[0][0],2))
    cosine_scores_caption_imgid[id_1]=[round(cosine_result[0][0],2),' '.join(reference),' '.join(caption_list)]
    
    #rescale cosine score for comparison with bleu
    rescale=np.arccos(cosine_result) / np.pi
    cosine_rescale_caption_imgid[id_1]=[round(rescale[0][0],2),' '.join(reference),' '.join(caption_list)]
    cosine_scores_rescale_list.append(round(rescale[0][0],2))  
    n+=1
    #print(i)
    #if i==4:
        #break
for score in cosine_scores_list:
    sum_score += score
for score1 in cosine_scores_rescale_list:
    sum_score_rescale += score1
final_avg_score=sum_score/n
final_avg_score_rescale=sum_score_rescale/n
print("Overall average is {:.2f}:".format(round(final_avg_score,2)))
print("Overall average after rescale is {:.2f}:".format(round(final_avg_score_rescale,2)))
#print(cosine_scores_list)
#print(cosine_scores_caption_imgid)
#print(cosine_rescale_caption_imgid)
#print(cosine_scores_rescale_list)


###############################computing average vector for predicted and reference captions#######
def calculate_avg_vector(text):
    embed_tensor = torch.tensor(())
    for word in text:
        try:
            tor1=torch.tensor([vocab.word2idx[word]], dtype=torch.long)
            embed_word=decoder.embed(tor1)
            embed_tensor=torch.cat([embed_tensor,embed_word],0)
        except:
            pass
    embed_tensor=torch.unsqueeze(torch.mean(embed_tensor,0),0)
    return embed_tensor.detach().numpy()


# In[133]:


##################2.3 Comparing text similarity methods##################
##################compare BLEU and cosine scores##########################

k=0
s=0
for key in cosine_scores_caption_imgid.keys():
    if key in cosine_rescale_caption_imgid.keys():
        bleu=str(cosine_rescale_caption_imgid[key][0])    #already rescaled
        cos=str(cosine_scores_caption_imgid[key][0]) 
        if bleu[0:3] == cos[0:3]:                         #checking till precision 1
            k+=1
            print("The imageid is :{0} ,matching score is :{1} and caption is :{2}".format(key,cosine_scores_caption_imgid[key][0],cosine_scores_caption_imgid[key][1]))
        if bleu[0:3] != cos[0:3]:
            s+=1
            print("The imageid is :{0}, score of Cosine and BLEU is :{1} & {2} and caption is :{3}".format(key,cosine_scores_caption_imgid[key][0],cosine_rescale_caption_imgid[key][0],cosine_scores_caption_imgid[key][1]))
print(k)
print(s)


# In[19]:


######## test data reference captions ###################
lines = read_lines(TOKEN_FILE_TEST)
test_image_ids, cleaned_captions = parse_lines(lines)

all_ref_pred = {}
z=0
for k,sent in enumerate(cleaned_captions):
    z=z+1
    img_id = test_image_ids[k]
    sent = word_tokenize(sent)
    if img_id in all_ref_pred:                
        inner_json = all_ref_pred[img_id]
        #print(inner_json)
        list1 = inner_json["original"]
        list1.append(sent)
        inner_json.update({'original': list1})
        all_ref_pred[img_id] = inner_json
    else:
        inner_json = {}
        list1 = []
        list1.append(sent)
        inner_json.update({'original': list1})
        all_ref_pred[img_id] = inner_json
print(len(all_ref_pred))





