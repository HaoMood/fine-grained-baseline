# -*- coding: utf-8 -*-
"""Training code.

Basic settings.
- Task: Fine-grained image classification.
- Dataset: CUB-200.
- Model: VGG-16 pre-trained on ImageNet.

Tricks.
- Two steps training.
"""


import collections
import os
import shutil
import tqdm

import torch
import torchvision
import visdom

import cub200
import model
import torchsummary


torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(32)


__all__ = ['TrainingManager']
__author__ = 'Hao Zhang'
__copyright__ = '2019 LAMDA'
__date__ = '2019-04-05'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 4.0'
__status__ = 'Development'
__updated__ = '2019-04-15'
__version__ = '2.6'


class TrainingManager():
    """Manager class for training.

    Attributes:
        _options: collections.
        _paths: collections[str].
        _model: torch.nn.Module.
        _loss_function: torch.nn.Module.
        _optimizer: collections[torch.optim.Optimizer].
        _scheduler: dict[torch.optim.lr_scheduler].
        _train_loader: torch.utils.data.DataLoader.
        _val_loader: torch.utils.data.DataLoader.
        _visdom: visdom.Visdom. Plotting tool.
    """
    def __init__(self, options, paths):
        """Model preparation."""
        # Configurations.
        self._options = options
        self._paths = paths

        # Prepare model and loss function.
        self._model = model.Model(architecture=self._options.arch,
                                  input_size=self._options.input_size,
                                  num_classes=options.num_classes)
        self._model = torch.nn.DataParallel(self._model).cuda()
        print(self._model)
        torchsummary.summary(
            self._model.module,
            input_size=(3, self._options.input_size, self._options.input_size))

        # Display configurations and options.
        print('PyTorch %s, CUDA %s, cuDNN %s, GPU %s' % (
            torch.__version__, torch.version.cuda,
            torch.backends.cudnn.version(), torch.cuda.get_device_name(0)))
        print(self._options)
        print('Code version %s' % __version__)
        print('-' * 80)

        # Prepare optimizer and optimizer scheduler.
        self._loss_function = torch.nn.CrossEntropyLoss().cuda()
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self._options.lr,
            momentum=0.9, weight_decay=self._options.weight_decay)
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, milestones=[60, 115], gamma=0.1)

        # Prepare dataset.
        resize_size = 512 if self._options.input_size == 448 else 256
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_size, resize_size)),
            torchvision.transforms.RandomCrop(
                (self._options.input_size, self._options.input_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_size, resize_size)),
            torchvision.transforms.CenterCrop(
                (self._options.input_size, self._options.input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
        ])
        train_data = cub200.CUB200(
            root=self._paths.data, train=True, transform=train_transform)
        val_data = cub200.CUB200(
            root=self._paths.data, train=False, transform=val_transform)
        print('Dataset size: train=%d, val=%d' % (len(train_data),
                                                  len(val_data)))
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options.bs,
            shuffle=True, num_workers=16, pin_memory=True)
        self._val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self._options.bs,
            shuffle=False, num_workers=16, pin_memory=True)

        # Visulization setup.
        self._visdom = visdom.Visdom(
            env='Term %d' % self._options.term, use_incoming_socket=False)
        assert self._visdom.check_connection()
        self._visdom.close()

    def train(self):
        """Train the model."""
        # Optionally resume from a checkpoint.
        start_epoch = 0
        end_epoch = self._options.epochs
        best_acc = 0.0
        if self._options.resume:
            checkpoint = torch.load(os.path.join(
                self._paths.model,
                'checkpoint_term_%d.tar.pth' % self._options.term))
            start_epoch = checkpoint['epoch'] - 1  # NOTE: t + 1 is stored
            best_acc = checkpoint['best_acc']
            self._model.load_state_dict(checkpoint['model'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            print('Load checkpoint_term_%d.tar.pth, at %d epoch.' % (
                self._options.term, start_epoch + 1))

        # Start training.
        print('           Train loss   Val. loss    Train acc.   '
              'Val. acc.    Learning rate')
        for t in range(start_epoch, end_epoch):
            self._scheduler.step()  # Learning rate schedule

            # Train and validation for one epoch.
            self._model.train()
            train_loss, train_acc = self._OneEpoch(t, is_train=True)
            self._model.eval()
            with torch.no_grad():
                val_loss, val_acc = self._OneEpoch(t, is_train=False)

            # Keep track of the best validation accuracy.
            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc
                print('*', end='')
            else:
                is_best = False
                print(' ', end='')

            # Save checkpoint.
            checkpoint = {
                'epoch': t + 1,  # NOTE: t + 1 is stored.
                'best_acc': best_acc,
                'model': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }
            src_filename = os.path.join(
                self._paths.model,
                'checkpoint_term_%d.tar.pth' % self._options.term)
            torch.save(checkpoint, src_filename)
            if is_best:
                shutil.copy(src_filename, os.path.join(
                    self._paths.model, 'best_model_term_%d.tar.pth' % (
                        self._options.term)))

            # Print the results.
            lr = [parameter_group['lr']
                  for parameter_group in self._optimizer.param_groups]
            print('Epoch %3d: %4.3f  %4.3f  %4.2f%%  %4.2f%%  %4.3e' %
                  (t + 1, train_loss, val_loss, train_acc, val_acc, lr[0]))
            self._visdomShow(
                t=t, train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc, val_acc=val_acc, lr=lr)

    def _OneEpoch(self, t: int, is_train: bool) -> (float, float):
        """Training for one epoch."""
        all_loss = []
        num_correct = 0
        num_total = 0
        for images, labels in tqdm.tqdm(
                self._train_loader if is_train else self._val_loader,
                desc=' Epoch %3d %s' % (t + 1,
                                        'train' if is_train else 'val. ')):
            # Data.
            images, labels = images.cuda(), labels.cuda()
            N = labels.size()[0]

            # Forward pass.
            score = self._model(images)
            loss = self._loss_function(score, labels)

            # Copmute training loss and accuracy.
            with torch.no_grad():
                all_loss.append(loss.item())
                prediction = torch.argmax(score, dim=1)
                num_correct += torch.sum(prediction == labels).item()
                num_total += N
            if not is_train:
                continue

            # Backward pass.
            self._optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     self._model.parameters(), max_norm=20)
            self._optimizer.step()
        return (sum(all_loss) / len(all_loss), 100 * num_correct / num_total)

    def _visdomShow(self, t: int, train_loss: float, val_loss: float,
                    train_acc: float, val_acc: float, lr: float):
        """Plot learninig curve."""
        plot_epoch = t + 1
        self._visdom.line(
            X=torch.Tensor([plot_epoch]), Y=torch.Tensor([train_loss]),
            name='train', win='Loss', update='append',
            opts={'xlabel': 'Epoch', 'ylabel': 'Loss', 'showlegend': True})
        self._visdom.line(
            X=torch.Tensor([plot_epoch]), Y=torch.Tensor([val_loss]),
            name='val', win='Loss', update='append',
            opts={'xlabel': 'Epoch', 'ylabel': 'Accuracy', 'showlegend': True})
        self._visdom.line(
            X=torch.Tensor([plot_epoch]), Y=torch.Tensor([train_acc]),
            name='train', win='Accuracy', update='append',
            opts={'xlabel': 'Epoch', 'ylabel': 'Accuracy', 'showlegend': True})
        self._visdom.line(
            X=torch.Tensor([plot_epoch]), Y=torch.Tensor([val_acc]),
            name='val', win='Accuracy', update='append',
            opts={'xlabel': 'Epoch', 'ylabel': 'Accuracy', 'showlegend': True})
        self._visdom.line(
            X=torch.Tensor([plot_epoch]), Y=torch.Tensor([lr[0]]),
            win='Learning rate', update='append',
            opts={'xlabel': 'Epoch', 'ylabel': 'lr', 'showlegend': True})


def main():
    """The main function."""
    # Parse command line arguments.
    import argparse
    parser = argparse.ArgumentParser(description='Training code.')
    parser.add_argument(
        '--arch', default='resnet50', type=str, help='Backbone model.')
    parser.add_argument(
        '--bs', default=64, type=int, help='Mini-batch size for training.')
    parser.add_argument(
        '--epochs', default=120, type=int, help='Epochs for training.')
    parser.add_argument(
        '--input_size', choices={224, 448}, default=448, type=int)
    parser.add_argument(
        '--lr', default=1e-2, type=float, help='Base lr for training.')
    parser.add_argument(
        '--num_classes', default=200, type=int, help='Number classes.')
    parser.add_argument(
        '--resume', action='store_true', help='Train from checkpoint.')
    parser.add_argument(
        '--term', type=int, default=1, help='Training program ID.')
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float, help='Weight decay.')
    args = parser.parse_args()

    # Useful paths and other configurations.
    project_root = os.path.dirname(os.getcwd())
    paths = collections.namedtuple('Paths', ['data', 'model'])(
        data=os.path.join('/data', 'share', 'data', 'cub200'),
        model=os.path.join(project_root, 'model'))
    for d in ['data', 'model']:
        assert os.path.isdir(getattr(paths, d))
    if args.resume:
        os.path.isfile(os.path.join(
            paths.model, 'checkpoint_term_%d.tar.pth' % args.term))

    # Start training.
    manager = TrainingManager(args, paths)
    manager.train()


if __name__ == '__main__':
    main()
