import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)    #1
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)                #2
    return tokenizer.decode(flat.tolist())


# A simple iterative generation loop for GPT2
def generate_text_simple(model, idx,                 #1 idx :  [batch, token_ids], max_new_tokens: how many new tokens needs to be generated
                         max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]    #2 Cropping based on the context size, selecting the last context elements
        with torch.no_grad():
            logits = model(idx_cond) # evaluate the model

        logits = logits[:, -1, :]                    #3 retrieving the logits of the last token
        probas = torch.softmax(logits, dim=-1)           #4
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)    #5 Finding the token id corresponding to the highest probs
        idx = torch.cat((idx, idx_next), dim=1)     #6 The new word is appended to the input for the next word generation

    return idx

# test if the model can produce coherent sentences
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device) # token ids are loaded into device
    with torch.no_grad(): # generate 50 new tokens for testing
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))      #1
    model.train() # set the model to train mode for the next batch


## Loss calculation via cross entropy loss
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)   # [batch, token, vocab]      #1 load data to GPU or CPU
    target_batch = target_batch.to(device)  # [batch, token]    
    logits = model(input_batch)
    ## CEL just takes the indices of the class and calculate the log(prob(that class))
    loss = torch.nn.functional.cross_entropy(  #[batch, token, vocab] >> [batch*token, vocab] for input batch
        logits.flatten(0, 1), target_batch.flatten() 
    )
    return loss


## Also calculate the total training and validation loss across all the batches
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)     #1
    else:
        num_batches = min(num_batches, len(data_loader))   #2
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()    #3
        else:
            break
    return total_loss / num_batches    # Average the loss across number of batches



# Calculate the train and validation losses averaged over batches
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  #1
    with torch.no_grad():                              #2
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


# plot the losses
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()                   #1
    ax2.plot(tokens_seen, train_losses, alpha=0)     #2
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()