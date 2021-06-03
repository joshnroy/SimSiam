def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calc_accuracy(classifier, dataloader, device, model=None):
    with torch.no_grad():
        if model is not None:
            model.eval()
        classifier.eval()
        acc_meter = AverageMeter('Accuracy')
        correct, total = 0, 0
        acc_meter.reset()
        all_predictions = []
        all_labels = []
        all_features = []
        for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = data[0]
            labels = data[-1]
            with torch.no_grad():
                if model is None:
                    preds = classifier(
                        images.to(device, non_blocking=True)).argmax(dim=1)
                else:
                    features = model(images.to(device, non_blocking=True))
                    preds = classifier(features)
                    all_features.append(features.cpu())
                correct = (preds == labels.to(
                    device, non_blocking=True)).sum().item()
                all_predictions.append(preds.cpu())
                all_labels.append(labels.cpu())
                acc_meter.update(correct/preds.shape[0])
        accuracy = acc_meter.avg
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_features = torch.cat(all_features, dim=0).numpy()

    return accuracy, all_labels, all_predictions, all_features

