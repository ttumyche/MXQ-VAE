import os
from datetime import datetime

import torch


def model_save(args, epoch, model, optimizer, loss):
    output_path = args.save_path + args.dataset + '_' + str(args.now)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    time = datetime.now().strftime('%m%e%H%M%S')
    model_path = os.path.join(output_path, f"Ep_{epoch}_{time}.pth")

    torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, model_path)

    print(f'Model saved to {model_path}')

    return model_path

def model_load(args, epoch, model, optimizer, loss):
    output_path = args.save_path + args.dataset + '_' + str(args.now)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    time = datetime.now().strftime('%m%e%H%M%S')
    model_path = os.path.join(output_path, f"Ep_{epoch}_{time}.pth")

    torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, model_path)

    return model_path