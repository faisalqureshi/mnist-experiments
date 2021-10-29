import os
import torch
from torch.utils import tensorboard

class Training:
    def __init__(self, device=-1):
        self.device = device # -1 suggests cuda isn't available
        self.current_epoch = 0
        self.writer = tensorboard.SummaryWriter()
        self.model = None
        self.training_dataloader = None
        self.validation_dataloader = None
        self.loss = None
        self.optimizer = None
        self.losses = {
            'train': [], 
            'validation': [],
            'epoch': []
        } 
        self.ave_time_per_epoch = {'train': 0, 'validation': 0}

        print(f'Training on {self.get_device_name()}')

    def set_device(self, device):
        self.device = device

    def get_device_name(self):
        if self.device == -1 or not torch.cuda.is_available():
            return 'CPU (No cuda-enabled device found)'
        else:
            return f'[{self.device}]: {torch.cuda.get_device_name(self.device)}'

    def gpu(self, o):
        """
        Ensure that the item isn't on gpu already.  Otherwise, I am not sure
        what is the effect of this call.
        """
        if self.device >= 0:
            return o.to(device=self.device)
        return o.to(device='cpu')

    def cpu(self, o):
        """
        Ensure that the item isn't on cpu already.  Otherwise, I am not sure
        what is the effect of this call.
        """
        if self.device >= 0:
            return o.to(device='cpu')
        return o

    def set_loss(self):
        self.loss = torch.nn.MSELoss()

    def set_optimizer(self, learning_rate, weight_decay):
        """
        Set model before setting up the optimizer.
        """
        assert(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def set_model(self, model):
        self.model = self.gpu(model)

    def get_model(self):
        return self.model

    def save_model(self, filepath):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict()
            }, filepath)

    def save_checkpoint(self, epoch):
        try:
            os.makedirs('./checkpoints')
        except:
            assert(os.path.isdir('./checkpoints'))

        filepath = f'./checkpoints/chkpt-{epoch:07}.pt'
        print(f'Saving checkpoint: {filepath} ...', end='')
        torch.save(
            {
                'loss': self.losses,
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'ave_time_per_epoch': self.ave_time_per_epoch
            }, 
            filepath
        )
        print(' done')

    def load_checkpoint(self, filepath):
        try:
            checkpoint = torch.load(filepath)
        except:
            print(f'[Failure] cannot read checkpoint file: {filepath}')
            return

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.losses['train'] = checkpoint['loss']['train']
            self.losses['validation'] = checkpoint['loss']['validation']
        
            # Backward compatibility
            try:
                self.ave_time_per_epoch = checkpoint['ave_time_per_epoch']
            except:
                print('[Warning] entry "ave_time_per_epoch" not found')
                self.ave_time_per_epoch = -1

            print(f'[Success] Checkpoint loaded: {filepath}')
        except:
            print(f'[Failure] cannot load checkpoint: {filepath}')

    def set_training_dataloader(self, training_dataloader):
        self.training_dataloader = training_dataloader

    def set_validation_dataloader(self, validation_dataloader):
        self.validation_dataloader = validation_dataloader

    def train(self, num_epoch, show_loss=False, checkpt_every=0):
        assert(self.model)
        assert(self.loss)
        assert(self.optimizer)
        assert(self.training_dataloader)

        start = torch.cuda.Event(enable_timing=True)    
        end = torch.cuda.Event(enable_timing=True)

        last_checkpt_epoch = -1

        phases = ['train']
        if self.validation_dataloader:
            phases.append('validation')

        timing_info = {'train': 0, 'validation': 0}
        for epoch in range(self.current_epoch+1, self.current_epoch+1+num_epoch):
            losses = {'train': 0.0, 'validation': 0.0}

            for phase in phases:
                if phase == 'train':
                    self.model.train()
                    dataloader = self.training_dataloader
                else:
                    self.model.eval()
                    dataloader = self.validation_dataloader

                start.record()

                n_batches = 0
                for data in dataloader:
                    n_batches += 1
                    imgs = self.gpu(data['sample'])

                    if phase == 'validation':
                        with torch.no_grad():
                            output = self.model(imgs)
                    else:
                        output = self.model(imgs)

                    batch_loss = self.loss(output, imgs)
                    losses[phase] += batch_loss.item()

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        batch_loss.backward()
                        self.optimizer.step()
            
                losses[phase] /= n_batches

                end.record()
                torch.cuda.synchronize()
                timing_info[phase] += start.elapsed_time(end)

            if show_loss:
                print(f'epoch={epoch:7}, train={losses["train"]:15.5}, validation={losses["validation"]:15.5}')
                # print(f'timing info={timing_info['train']}/{timing_info['validation']}')
            self.writer.add_scalar("Loss/train", losses["train"], epoch)
            self.writer.add_scalar("Loss/validation", losses["validation"], epoch)
            
            self.losses['epoch'].append(epoch)
            for phase in phases:
                self.losses[phase].append(losses[phase])

            if checkpt_every > 0 and epoch % checkpt_every == 0:
                for phase in phases:
                    self.ave_time_per_epoch[phase] = timing_info[phase] / max(epoch - self.current_epoch, 1)
                self.save_checkpoint(epoch)
                last_checkpt_epoch = epoch

        for phase in phases:
            self.ave_time_per_epoch[phase] = timing_info[phase] / num_epoch
        print('Average time per epoch (ms):')
        print(f'\ttrain={self.ave_time_per_epoch["train"]:16.4}, validation={self.ave_time_per_epoch["validation"]:16.4}')

        self.current_epoch = epoch   # Next time training starts at self.current_epoch+1

        if last_checkpt_epoch != epoch:
            self.save_checkpoint(epoch)       # Save at the end of this training run

