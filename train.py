"""
$ python train.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import tiktoken
from model import GPT #, GPTConfig
from dataclasses import dataclass

from muon_adam_polar import SingleDeviceMuonWithAuxAdam

# -----------------------------------------------------------------------------
@dataclass
class TrainConfig():
    eval_interval = 2000
    eval_steps = 200
    out_dir = 'out'
    eval_only = False 
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    begin_with_eval = False

    # logging
    log_interval = 1
    log_time = True
    wandb_log = True
    wandb_project = 'optim_experiments'
    wandb_run_name = 'nanoGPT-muon-1' # 'run' + str(time.time())
    print_log = True
    log_input_text = True
    log_text_interval = 200
    log_text_length = 400

    # data
    dataset = 'openwebtext'
    gradient_accumulation_steps = 4 
    batch_size = 12 
    block_size = 2048

    # model
    n_layer = 12
    n_head = 12
    d_model = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False 
    vocab_size = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    is_inference = False
    compile = True

    # muon and adamw optimizer
    max_steps = 600000 # total number of training steps
    adam_max_lr = 3e-4 # max learning rate
    adam_min_lr = 1e-5 #  should be ~= learning_rate/10 per Chinchilla
    muon_lr = 0.02
    weight_decay = 1e-1
    betas = (0.9, 0.95)
    momentum = 0.95
    eps = 1e-10
    grad_clip = 1.0 # disable if == 0.0
    warmup_steps = 2000 
    lr_decay_steps = 600000 # should be ~= max_steps per Chinchilla

    # training
    cce = True
    training = True

    # system
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 

# -----------------------------------------------------------------------------
def get_batch(args, split, data_dir, device_type):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(args.device, non_blocking=True), y.pin_memory().to(args.device, non_blocking=True)
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y

def configure_optimizers(model, args):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    muon_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    adamw_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': muon_params, 'use_muon': True, 'lr': args.muon_lr, 'momentum': args.muon_lr, 'weight_decay': args.weight_decay},
        {'params': adamw_params,'use_muon': False, 'lr': args.adam_max_lr, 'betas': args.betas, 'eps': args.eps, 'weight_decay': 0.0}
    ]
    n_params_muon = sum(p.numel() for p in muon_params)
    n_params_adamw = sum(p.numel() for p in adamw_params)
    print(f"muon params (2d): {n_params_muon/1e6}")
    print(f"adamw params (2d): {n_params_adamw/1e6}")

    muon_adamw_optimizer = SingleDeviceMuonWithAuxAdam(param_groups=optim_groups)
    return muon_adamw_optimizer

# -----------------------------------------------------------------------------
def main(train_args):
    args = train_args
    # seed_offset = 1
    # torch.manual_seed(1337 + seed_offset)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    data_dir = os.path.join('data', args.dataset)

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        block_size=args.block_size,
        d_model=args.d_model, 
        bias=args.bias,
        vocab_size=None, 
        dropout=args.dropout
    ) # start with model_args from command line

    try:
        if args.init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
            model = GPT(args)

        elif args.init_from == 'resume':
            print(f"Resuming training from {args.out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'd_model', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            model = GPT(args)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            step = checkpoint['step']
            best_val_loss = checkpoint['best_val_loss']


        # crop down the model block size if desired, using model surgery
        if args.block_size < model.config.block_size:
            model.crop_block_size(args.block_size)
            model_args['block_size'] = args.block_size # so that the checkpoint will have the right value
        model.to(args.device).to(dtype=ptdtype)

        # optimizer
        optimizer = configure_optimizers(model, args)
        if args.init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory

        # compile the model
        if args.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model)

        # logging
        if args.wandb_log:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=model_args)

        # -----------------------------------------------------------------------------
        X, Y = get_batch(args, 'train', data_dir, device_type) # fetch the very first batch
        if args.log_time: 
            t0 = time.time() 
            training_start_time = time.time()
        raw_model = model # unwrap DDP container if needed
        enc = tiktoken.get_encoding("gpt2")
        running_mfu = -1.0
        tokens_per_step = args.gradient_accumulation_steps * args.batch_size * args.block_size
        print(f"tokens per macro-step will be: {tokens_per_step:,}")

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        step = 0
        best_val_loss = 1e9
        trained_token_count = 0

        while True:

            # validation 
            # -----------------------------------------------------------------------------
            if step % args.eval_interval == 0 and (step != 0 or args.begin_with_eval):
                with torch.no_grad():
                    losses = {}
                    model.eval()
                    for split in ['train', 'val']:
                        batch_losses = torch.zeros(args.eval_steps)
                        for k in range(args.eval_steps):
                            X, Y = get_batch(args, split, data_dir, device_type)
                            if args.cce:
                                loss = model(X, Y)
                            else:
                                logits, loss = model(X, Y)
                            batch_losses[k] = loss.item()
                        losses[split] = batch_losses.mean()
                    model.train()

                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if args.wandb_log:
                    wandb.log({
                        "step": step,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "muon_lr": args.muon_lr,
                        "adamw_lr": args.adam_max_lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })
                if losses['val'] < best_val_loss or args.always_save_checkpoint:
                    best_val_loss = losses['val']
                    if step > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'step': step,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        print(f"saving checkpoint to {args.out_dir}")
                        torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
            if step == 0 and args.eval_only:
                break

            # training
            # -----------------------------------------------------------------------------
            for micro_step in range(args.gradient_accumulation_steps):
                with ctx:
                    # cce doesn't materialize raw logits
                    if args.cce:
                        loss = model(X, Y)
                    else:
                        logits, loss = model(X, Y)
                    if micro_step == 0 and step % args.log_text_interval == 0:
                        print("\n----- GROUND-TRUTH -----")
                        print(enc.decode((Y[0]).tolist())[:300], "\n")  # First 100 chars
                        if not args.cce and 'logits' in locals():
                            print("\n----- PREDICTED -----")
                            print(enc.decode((logits[0,:,:50257].argmax(dim=-1)).tolist())[:300])
                        print("-" * 40)
                    loss = loss / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch(args, 'train', data_dir, device_type)
                # backward pass, with gradient scaling if training in fp16
                loss.backward()
            # clip the gradient
            if args.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            trained_token_count += tokens_per_step


            # timing and logging
            # -----------------------------------------------------------------------------
            if step % args.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                if args.print_log or args.wandb_log:
                    lossf = loss.item() * args.gradient_accumulation_steps
                if args.log_time:
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                    time_str = f"{dt*1000:.2f}" 
                    tokens_per_sec = tokens_per_step / dt
                else:
                    dt = None
                    tokens_per_sec = "N/A"
                    time_str = "N/A"
                if args.print_log:
                    print(f"step {step}: loss {lossf:.4f}, time {time_str} ms/step, {tokens_per_sec:.2f} tokens/s, mfu {running_mfu*100:.2f}%") 
                if args.wandb_log: 
                    wandb.log({
                        "optim_step": step // args.gradient_accumulation_steps,
                        "train/loss": lossf,
                        "muon_lr": args.muon_lr,
                        "adamw_lr": args.adam_max_lr,
                        "mfu": running_mfu*100, # convert to percentage
                        # **({'elapsed_time' : time.time() - training_start_time} if args.log_time else {}) ,
                        "tokens" : trained_token_count,
                        'tokens_per_sec' : tokens_per_sec,
                        'perplexity' : torch.exp(lossf)
                    })
            step += 1

            # termination conditions
            if step > args.max_steps:
                break

    except(KeyboardInterrupt):
        print("key exit")
                
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'step': step,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving checkpoint to {args.out_dir}")
    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    # dynamo.config.capture_dynamic_output_shape_ops = True
    train_config = TrainConfig()
    print(train_config)
    main(train_config)

