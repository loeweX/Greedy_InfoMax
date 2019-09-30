import torch
import time

def val_by_CPC(opt, model, test_loader):
    total_step = len(test_loader)

    loss_epoch = [0 for i in range(opt.model_splits)]
    starttime = time.time()

    for step, (audio, _, _, _) in enumerate(test_loader):

        model_input = audio.to(opt.device)

        loss = model(model_input)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x/total_step for x in loss_epoch]
    return validation_loss
