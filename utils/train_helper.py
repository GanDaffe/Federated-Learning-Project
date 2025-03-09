def train(net, 
          trainloader, 
          criterion, 
          optimizer, 
          device, 
          proximal_mu: float = None):
    
    net.train()
    running_loss, running_corrects, tot = 0.0, 0, 0

    global_params = copy.deepcopy(net).parameters()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels)

        if proximal_mu is not None:
            proximal_term = sum((local_weights - global_weights).norm(2) 
                                for local_weights, global_weights in zip(net.parameters(), global_params))
            loss += (proximal_mu / 2) * proximal_term

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        tot += images.shape[0] 

        running_corrects += torch.sum(predicted == labels).item()
        running_loss += loss.item() * images.shape[0]

        del images, labels, outputs, loss, predicted

    running_loss /= tot
    accuracy = running_corrects / tot

    del global_params, tot
    torch.cuda.empty_cache()
    gc.collect()

    return running_loss, accuracy
