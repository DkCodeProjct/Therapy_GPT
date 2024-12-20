# Therapy_GPT
  Therapy GPT like transformer model


   # However This is My First Chat Bot So, Id Like to Do Mistakes As much as Possible
   # And will Continue to Improve This model
   
![Screenshot](Screenshot%20from%202024-12-20%2010-42-24.png)


### Im quit happy with the result, 
   * these are the hyper para ive used to train the model
        
          batchsiz = 64
          blocksiz = 128
          epochs = 700
          evalIntervals = 100
          lr = 3e-4
          device = "cuda" if torch.cuda.is_available() else "cpu"
          evaliters = 200
          nemb = 158
          nhead = 4
          nlayers = 4
          dropout = 0.2

## Now i don't know if i did the Fine tune stage or Not:
   * Cos, I wrote this model on Andrej karpathy [Lets reproduce GPT2]
   * and He didnt show how to Finetune
   * So of Course IVe been slept with GPT and Ask it to Wrote me Fine-tune Code

## Ive Learn So many things With this model:
  * I got Good grasp on Model architecture
  * Training, val/dev losses
  * Hyper para
  
  * **Specialy About The Tokenization**
      - so i use  | "HuggingFaceTB/SmolLM2-1.7B-Instruct" |
      - Before i use GPT2 Tokenizer which has Vocab siz of 50257
      - So i ask myself So if Tokenizer have 50257 Vocab itll take Long Time to train
      - So ive switch to This tokenizer which has <vocab size that GPT2
      - indeed it Took less time to train

### Also im not Quite Sure About the model Save Neither>
        
          def saveCheckpnt(model, optimizer, epoch, loss, filepath):
              checkPnt = {
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch,
                  "loss": loss,
              }
              torch.save(checkPnt, filepath)
              print(f"Checkpoint saved to {filepath}")

